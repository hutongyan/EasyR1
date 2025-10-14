# Off-Policy Speculative Guidance in EasyR1

This note documents the minimal patches we applied on top of upstream EasyR1 to support token-level off-policy guidance with speculative decoding. The goal is to let a small draft model propose tokens, while a stronger guidance model can override them per step; we then use importance sampling to update the draft model with guidance-assisted rollouts.

## Rollout extensions

- Added `OffPolicyGuidanceConfig` to `verl/workers/rollout/config.py` and exposed it from `verl/workers/rollout/__init__.py` so runs can supply a guidance checkpoint, tensor-parallel width, sampling overrides, and number of requested logprobs.
- `verl/workers/rollout/vllm_rollout_spmd.py` now instantiates an optional second `LLM` engine when `offpolicy_guidance` is enabled. During generation it queries both engines with the same prompts, aligns their outputs token-wise, and records:
  - `offpolicy_mask`: Boolean tensor marking tokens that came from the guidance model rather than the draft actor.
  - `guidance_log_probs`: Per-token logprobs reported by the guidance engine (falling back to draft logprobs if missing).
- The combined batch returned to the trainer therefore contains the original prompts/responses plus the new tracking tensors, all properly padded and repeated when `n > 1`.

## Trainer changes

- In `verl/trainer/ray_trainer.py` the driver now looks for the new fields after recomputing `old_log_probs`. When present it:
  - Builds `behavior_log_probs` by mixing draft and guidance logprobs according to `offpolicy_mask`.
  - Computes importance weights `exp(old_log_probs - behavior_log_probs)` with configurable clamping and optional gradient detachment (new knobs in `AlgorithmConfig`).
  - Logs guidance diagnostics such as off-policy token ratio, sequence ratio, weight statistics, and missing-logprob ratio.
- The actor worker (`verl/workers/actor/dp_actor.py`) passes `behavior_log_probs` and `importance_weights` into the PPO minibatch updates so every micro-batch has access to the corrected ratios.
- `compute_policy_loss` in `verl/trainer/core_algos.py` accepts the optional tensors, switching the PPO ratio numerator to use the behavior policy and multiplying advantages by the provided importance weights before clipping.

## Configuration knobs

- `AlgorithmConfig` gained `importance_clamp_min`, `importance_clamp_max`, and `importance_detach` so experiments can tune weight stability without touching code.
- `RolloutConfig` exposes the nested `offpolicy_guidance` block; leaving it `None` reproduces the original EasyR1 behaviour, so the feature is opt-in.

## Expected workflow

1. Point `worker.rollout.offpolicy_guidance.model_path` at the guidance checkpoint and set `enable=True`. Make sure both models share a tokenizer since outputs are aligned by token id.
2. Keep the actor rollout engine pointing at the smaller draft checkpoint. During rollout, the guidance engine will run speculatively without affecting the draft’s logprob recomputation step.
3. Train as usual. The trainer will log `guidance/*` metrics so you can monitor how many tokens come from the guidance model and how the importance weights behave.

With these changes you can mix actor-generated and guidance-generated tokens inside the same batch, while the PPO update remains numerically stable via importance sampling. Removing or disabling `offpolicy_guidance` returns the code to its vanilla EasyR1 behaviour.
