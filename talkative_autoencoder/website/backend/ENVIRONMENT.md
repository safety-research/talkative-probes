## Environment configuration and precedence

- .env loading order:
  - `website/backend/.env` is loaded first
  - repo-root `.env` is loaded second with `override=false` (backend/.env wins)

- Infrastructure settings (read by `load_settings()` into `Settings`):
  - Examples: `DEVICE`, `DEVICES`, `NUM_WORKERS_PER_GPU`, `ALLOWED_ORIGINS`, `API_KEY`, `MAX_QUEUE_SIZE`, `MAX_TEXT_LENGTH`, `HOST`, `PORT`, `REQUEST_TIMEOUT`, `RATE_LIMIT_PER_MINUTE`, `TUNED_LENS_DIR`, `LAZY_LOAD_MODEL`, `MAX_CPU_CACHED_MODELS`, `DEFAULT_GROUP`, etc.
  - Types: integers, booleans, comma lists handled in `config.load_settings()`.
  - Startup default group precedence (only this mechanism controls startup default):
    1) `DEFAULT_GROUP` env
    2) `model_groups.json` → `settings.default_group` (keep this in JSON for sensible defaults)
    3) No default

- Model/group runtime overrides (applied when loading groups/models via `Settings.get_model_config_with_overrides()`):
  - Allowed settings to override: `use_bf16`, `no_orig`, `no_kl`, `batch_size`, `auto_batch_size_max`, `comparison_tl_checkpoint`, `preload_groups`, `max_cpu_models`
  - Precedence for a given setting:
    1) `MODEL_<model_id>_<SETTING>`
    2) `MODEL_GROUP_<group_id>_<SETTING>`
    3) `GLOBAL_<SETTING>`
    4) `model_groups.json` `settings.<setting>`
  - Note: `default_group` is NOT overridable via `GLOBAL_*`/`MODEL_GROUP_*`/`MODEL_*`. Use `DEFAULT_GROUP` env only.
  - Hyphens in IDs are normalized to underscores for env var names. Example:
    - model id `gemma3-27b-chat` → `MODEL_gemma3_27b_chat_BATCH_SIZE=16`
    - group id `gemma2-9b-it` → `MODEL_GROUP_gemma2_9b_it_BATCH_SIZE=32`
  - Examples:
    - Global default batch size: `GLOBAL_BATCH_SIZE=64`
    - Group-specific: `MODEL_GROUP_gemma2_9b_it_BATCH_SIZE=32`
    - Model-specific: `MODEL_gemma3_27b_chat_BATCH_SIZE=16`

- JSON settings (`website/backend/app/model_groups.json`):
  - Top-level `settings`: `preload_groups`, `default_group`, `max_cpu_models`, `use_bf16`, `no_orig`, `no_kl`, `comparison_tl_checkpoint`, `estimated_gpu_memory`
  - `max_cpu_models` controls CPU cache size. If not overridden, falls back to `MAX_CPU_CACHED_MODELS` infra env via `Settings.max_cpu_cached_models`.

- Quick examples for `.env`:
```env
# Startup default group
DEFAULT_GROUP=gemma3-27b-it

# Infrastructure
DEVICES=cuda:0,cuda:1
NUM_WORKERS_PER_GPU=1
ALLOWED_ORIGINS=http://localhost:3000
RATE_LIMIT_PER_MINUTE=60
LAZY_LOAD_MODEL=false

# Global model/group behavior
GLOBAL_USE_BF16=true
GLOBAL_NO_ORIG=true
GLOBAL_BATCH_SIZE=48
GLOBAL_AUTO_BATCH_SIZE_MAX=768

# Group-specific override
MODEL_GROUP_gemma2_9b_it_BATCH_SIZE=64

# Model-specific override
MODEL_gemma3_27b_chat_BATCH_SIZE=16
```


