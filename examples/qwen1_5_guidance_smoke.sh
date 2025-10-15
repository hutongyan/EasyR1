#!/bin/bash

set -euo pipefail

# Small supported checkpoints from the Qwen2.5 family.
DRAFT_MODEL=${DRAFT_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
GUIDANCE_MODEL=${GUIDANCE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}

EXP_NAME=${EXP_NAME:-qwen15_guidance_smoke}

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.rollout_batch_size=4 \
    data.mini_rollout_batch_size=4 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    worker.actor.global_batch_size=4 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.model.model_path=${DRAFT_MODEL} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.n=1 \
    worker.rollout.gpu_memory_utilization=0.4 \
    worker.rollout.offpolicy_guidance.enable=true \
    worker.rollout.offpolicy_guidance.model_path=${GUIDANCE_MODEL} \
    worker.rollout.offpolicy_guidance.tensor_parallel_size=1 \
    worker.rollout.offpolicy_guidance.logprobs=4 \
    algorithm.importance_clamp_min=0.1 \
    algorithm.importance_clamp_max=5.0 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    trainer.logger='[console]' \
    trainer.max_steps=1 \
    trainer.val_before_train=false \
    trainer.val_freq=-1 \
    trainer.save_freq=-1 \
    trainer.save_checkpoint_path=null \
    trainer.experiment_name=${EXP_NAME}
