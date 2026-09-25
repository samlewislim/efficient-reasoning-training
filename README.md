# Length-controlled reasoning with GRPO

Training code for three methods that shorten a reasoning model's chain of
thought, all trained with GRPO + LoRA on top of the same base models and the
same data.

| Method | Reward | Length signal |
|---|---|---|
| **ThinkPrune** | `1[correct(clip(y, L))]` | hard generation clip at `L` |
| **Kimi k1.5 (GLP)** | `1[correct(y)] + w · len_reward(y)` | group-relative, no prompt instruction |
| **LCPO-Exact** | `1[correct(y)] − α·\|n_gold − n_y\|` | target length written into the prompt |

## Layout

```
train_thinkprune.py    ThinkPrune
train_kimi.py          Kimi k1.5 length penalty
train_lcpo_exact.py    LCPO-Exact
common.py              shared CLI, GRPOConfig, LoRA, callbacks
data.py                training / evaluation datasets and prompt formats
grading.py             answer extraction and correctness scoring
vendor/qwen_math.py    Qwen2.5-Math grader (vendored, unmodified)
aux_data/              AIME evaluation set (vendored from the ThinkPrune release)
```

Each `train_*.py` supplies only its reward function, its prompt formatters,
and any method-specific arguments; everything else is shared.

## Running

```bash
accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 \
    train_thinkprune.py \
        --model_name Qwen/Qwen3-8B \
        --max_completion_length 4000 \
        --num_generations 16 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 128 \
        --lora_rank 16 \
        --output_dir outputs/thinkprune-8b
```

`pdbs × grad_accum × world_size ÷ num_generations` is the number of prompts
per optimizer step; the configuration above gives 32.

Method-specific flags:

- `train_thinkprune.py` — `--max_completion_length` is the budget `L`
- `train_kimi.py` — `--length_weight`, `--length_penalty_warmup_steps`
- `train_lcpo_exact.py` — `--n_min`, `--n_max`, `--alpha`

## Requirements

`trl`, `transformers`, `peft`, `accelerate`, `datasets`, `vllm`, `torch`,
`numpy`, plus `sympy` / `latex2sympy2_extended` / `word2number` for the
grader. Generation uses TRL's colocated vLLM mode.

## Attribution

- `vendor/qwen_math.py` — Qwen2.5-Math grader, unmodified.
- `aux_data/test_aime.parquet` — from the ThinkPrune release (Apache 2.0),
  <https://github.com/UCSB-NLP-Chang/ThinkPrune>.
- Training data: [PRIME-RL/Eurus-2-RL-Data](https://huggingface.co/datasets/PRIME-RL/Eurus-2-RL-Data),
  `numina_amc_aime` subset.

Method references: ThinkPrune (arXiv:2504.01296), Kimi k1.5
(arXiv:2501.12599 §2.3.3), LCPO-Exact (arXiv:2503.04697 Eq. 1).
