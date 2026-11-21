# SafeCal small open-weight language-model rollout smoke

**Date:** 21 November 2026

**Execution:** [GitHub Actions run 44873611480](https://github.com/Venkat-RJ/openresearch/actions/runs/44873611480) using the actual installed Verifiers v1 package, `SafeCalTaskset`, and `SafeCalVerifierEnv.run`. Model-generated dialogue was produced through local Hugging Face Transformers CPU inference with a custom interaction wrapper, **not** the native `vf-eval` model-provider client. Model: `Qwen/Qwen2.5-0.5B-Instruct`, greedy decoding, up to 96 new tokens per turn.

**Result:** Two real model-generated episodes executed in the environment. Neither completed a valid calibration. Both scored zero because the agent broke the one-action JSON protocol.

| Split | First action | Second action | Third action | Completed | Reward |
|---|---|---|---|---:|---:|
| Single fault | Valid `reference_check` | Repeated `reference_check` with an extra JSON field | Natural-language request for more information | No | 0.0 |
| Combined fault | Valid `reference_check` | Repeated `reference_check` with an extra JSON field | Natural-language request for more information | No | 0.0 |

Each episode spent 1.5 normalized diagnostic cost units, then hit the two-invalid-action limit. The model appears to have copied the measurement into an invalid action.

These are smoke tests, **not** a representative evaluation of language-model capability. A proper baseline needs a larger model, grammar-constrained actions, many independent tasks, and preferably the native `vf-eval` model interface. This is not language-model RL training.

The separate [Verifiers v1 compatibility run](https://github.com/Venkat-RJ/openresearch/actions/runs/44873170003) also passed. The small CPU policy-gradient training pilot is documented in [RL_PILOT.md](RL_PILOT.md). No cloud GPUs or real actuator were involved.
