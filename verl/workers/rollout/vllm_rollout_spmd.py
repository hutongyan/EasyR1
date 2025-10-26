# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from contextlib import contextmanager
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import OffPolicyGuidanceConfig, RolloutConfig


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    # repeat the elements, supports both tensor and numpy array
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    # enforce vllm to not output image token
    # TODO: add video token
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any], min_pixels: int, max_pixels: int, video_fps: float
) -> dict[str, Any]:
    # may convert image path to image object
    images, videos = [], []
    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        for video in multi_modal_data["videos"]:
            videos.append(process_video(video, min_pixels, max_pixels, video_fps))

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    return None


def _resolve_logprob_entry(entry: Any, token_id: int) -> float:
    if entry is None:
        return float("nan")

    # vLLM may return a dict mapping token ids/strings to floats
    if isinstance(entry, dict):
        value = entry.get(token_id)
        if value is None:
            value = entry.get(str(token_id))
        if value is None and hasattr(entry, "items"):
            for key, candidate in entry.items():
                if hasattr(candidate, "token_id") and candidate.token_id == token_id:
                    value = candidate
                    break
        if hasattr(value, "logprob"):
            return float(value.logprob)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    # some versions return an iterable of candidates
    try:
        iterator = iter(entry)
    except TypeError:
        return float("nan")

    for candidate in iterator:
        cand_token = getattr(candidate, "token_id", getattr(candidate, "id", None))
        if cand_token is None and isinstance(candidate, tuple) and len(candidate) >= 2:
            cand_token = candidate[0]
            candidate_value = candidate[1]
        else:
            candidate_value = getattr(candidate, "logprob", None)

        if cand_token == token_id or str(cand_token) == str(token_id):
            if candidate_value is None and isinstance(candidate, tuple) and len(candidate) >= 2:
                candidate_value = candidate[1]

            if candidate_value is not None:
                try:
                    return float(candidate_value)
                except (TypeError, ValueError):
                    return float("nan")

    return float("nan")


def _extract_sequence_logprobs(sequence_output: Any) -> list[float]:
    # sequence_output.logprobs is a per-token structure.
    logprob_entries = getattr(sequence_output, "logprobs", None)
    token_ids = getattr(sequence_output, "token_ids", [])
    if logprob_entries is None:
        return [float("nan")] * len(token_ids)

    chosen = []
    for token_id, entry in zip(token_ids, logprob_entries):
        chosen.append(_resolve_logprob_entry(entry, token_id))

    return chosen


def _merge_guidance_outputs(draft_output: Any, guidance_output: Any) -> tuple[list[int], list[int], list[float]]:
    guidance_tokens = getattr(guidance_output, "token_ids", [])
    draft_tokens = getattr(draft_output, "token_ids", [])

    offpolicy_mask = []
    for idx, token_id in enumerate(guidance_tokens):
        is_offpolicy = idx >= len(draft_tokens) or draft_tokens[idx] != token_id
        offpolicy_mask.append(1 if is_offpolicy else 0)

    guidance_logprobs = _extract_sequence_logprobs(guidance_output)
    return guidance_tokens, offpolicy_mask, guidance_logprobs


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        engine_kwargs = {}
        if processor is not None:  # only VLMs have processor
            engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        self.guidance_config: Optional[OffPolicyGuidanceConfig] = (
            config.offpolicy_guidance if config.offpolicy_guidance and config.offpolicy_guidance.is_enabled() else None
        )

        speculative_config = None
        if self.guidance_config is not None:
            speculative_config = {"model": self.guidance_config.model_path}
            if self.guidance_config.tensor_parallel_size is not None:
                speculative_config["tensor_parallel_size"] = self.guidance_config.tensor_parallel_size
            if self.guidance_config.num_speculative_tokens is not None:
                speculative_config["num_speculative_tokens"] = self.guidance_config.num_speculative_tokens
            if self.guidance_config.sampling_overrides:
                speculative_config.update(self.guidance_config.sampling_overrides)
            enable_sleep_mode = False
        else:
            enable_sleep_mode = True

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=enable_sleep_mode,
            speculative_config=speculative_config,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage when sleep mode is enabled
        if enable_sleep_mode:
            self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)
        sampling_kwargs["n"] = 1
        sampling_kwargs["logprobs"] = max(1, sampling_kwargs.get("logprobs", 0))
        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

        self.guidance_sampling_params: Optional[SamplingParams] = None

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        old_guidance_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
                if self.guidance_sampling_params is not None and hasattr(self.guidance_sampling_params, key):
                    old_value = getattr(self.guidance_sampling_params, key)
                    old_guidance_params_args[key] = old_value
                    setattr(self.guidance_sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)
        for key, value in old_guidance_params_args.items():
            setattr(self.guidance_sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)

        target_n = max(1, getattr(self.config, "n", 1))
        if target_n < 1:
            target_n = 1

        if target_n > 1:
            input_ids = _repeat_interleave(input_ids, target_n)
            attention_mask = _repeat_interleave(attention_mask, target_n)
            position_ids = _repeat_interleave(position_ids, target_n)
            if batch_multi_modal_data is not None:
                batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, target_n)
            batch_raw_prompt_ids = [raw_prompt_id for raw_prompt_id in batch_raw_prompt_ids for _ in range(target_n)]
            batch_size = input_ids.size(0)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        processed_mm: list[Optional[dict[str, Any]]] = []
        if batch_multi_modal_data is not None:
            for multi_modal_data in batch_multi_modal_data:
                processed_mm.append(
                    _process_multi_modal_data(
                        multi_modal_data,
                        prompts.meta_info["min_pixels"],
                        prompts.meta_info["max_pixels"],
                        prompts.meta_info["video_fps"],
                    )
                )
        if not processed_mm:
            processed_mm = [None] * batch_size

        vllm_inputs = []
        for raw_prompt_ids, mm_data in zip(batch_raw_prompt_ids, processed_mm):
            entry = {"prompt_token_ids": list(raw_prompt_ids)}
            if mm_data is not None:
                entry["multi_modal_data"] = mm_data
            vllm_inputs.append(entry)

        with self.update_sampling_params(**prompts.meta_info):
            completions: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm
            )

        response_sequences: list[list[int]] = []
        offpolicy_masks: list[list[int]] = []
        guidance_logprob_list: list[list[float]] = []
        spec_metrics: list[Any] = []

        for completion in completions:
            metrics = getattr(completion, "metrics", None)
            for output in completion.outputs:
                tokens = list(output.token_ids)
                response_sequences.append(tokens)
                offpolicy_masks.append([1] * len(tokens))  # treat all tokens as off-policy
                guidance_logprob_list.append(_extract_sequence_logprobs(output))
                spec_metrics.append(metrics)

        response_ids = VF.pad_2d_list_to_length(
            response_sequences, self.pad_token_id, max_length=self.config.response_length
        ).to(input_ids.device)
        offpolicy_mask_tensor = torch.ones_like(response_ids, dtype=torch.bool)
        guidance_logprob_tensor = VF.pad_2d_list_to_length(
            guidance_logprob_list, float("nan"), max_length=self.config.response_length
        ).to(input_ids.device)
        guidance_logprob_tensor = guidance_logprob_tensor.to(dtype=torch.float32)
        batch_size = response_ids.size(0)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:  # qwen2vl mrope: (batch_size, 4, seq_length)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if offpolicy_mask_tensor is not None:
            batch["offpolicy_mask"] = offpolicy_mask_tensor
            batch["guidance_log_probs"] = guidance_logprob_tensor
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}
        if spec_metrics:
            non_tensor_batch["speculative_metrics"] = np.array(spec_metrics, dtype=object)

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)
