"""LCPO-Exact (arXiv:2503.04697, Eq. 1): length-conditioned policy optimisation.

    r(y, n_gold) = 1[correct(y)] - alpha * |n_gold - n_y|

A target length n_gold is sampled uniformly from [n_min, n_max] per prompt and
written into the user turn as "Think for {n_gold} tokens.". The model is
penalised linearly for deviating from it in either direction.

Unlike ThinkPrune there is no per-example hard clip: n_gold sits well below
the global generation cap, so the model is free to overshoot or undershoot and
the penalty term -- not truncation -- is what teaches it to hit the target.

This produces a length-conditioned model, so it is evaluated across a range of
requested budgets rather than at one operating point.

Usage:
    accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 \
        train_lcpo_exact.py --model_name Qwen/Qwen3-8B --n_min 100 --n_max 4000
"""
import numpy as np
from transformers import AutoTokenizer

from common import base_arg_parser, run
from data import eval_question, ground_truth, user_question
from grading import completion_text, math_score


def parse_args():
    p = base_arg_parser(__doc__.splitlines()[0])
    p.add_argument("--n_min", type=int, default=100,
                   help="Lower bound of the sampled target length.")
    p.add_argument("--n_max", type=int, default=4000,
                   help="Upper bound of the sampled target length.")
    p.add_argument("--alpha", type=float, default=0.0003,
                   help="Weight on the absolute length deviation.")
    p.add_argument("--target_seed", type=int, default=123,
                   help="Seeds the per-prompt target lengths, independently of --seed.")
    return p.parse_args()


args = parse_args()
# Bound at module level because the reward function below reads it as a
# global (see LengthPenaltyState's note on functools.partial).
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
rng = np.random.default_rng(args.target_seed)

LENGTH_INSTRUCTION = "\nThink for {n} tokens."


def lcpo_exact_reward(completions, solution, n_gold, **kwargs) -> list[float]:
    rewards = []
    for completion, sol, target in zip(completions, solution, n_gold):
        text = completion_text(completion)
        n_y = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        rewards.append(math_score(text, sol) - args.alpha * abs(target - n_y))
    return rewards


def _sample_target() -> int:
    return int(rng.integers(args.n_min, args.n_max + 1))


def format_train(example):
    n_gold = _sample_target()
    return {
        "prompt": [{"role": "user",
                    "content": user_question(example) + LENGTH_INSTRUCTION.format(n=n_gold)}],
        "solution": ground_truth(example),
        "n_gold": n_gold,
    }


def format_eval(example):
    n_gold = _sample_target()
    return {
        "prompt": [{"role": "user",
                    "content": eval_question(example) + LENGTH_INSTRUCTION.format(n=n_gold)}],
        "solution": ground_truth(example),
        "n_gold": n_gold,
    }


if __name__ == "__main__":
    run(args, lcpo_exact_reward, format_train, format_eval, tokenizer)
