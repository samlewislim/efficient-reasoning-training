"""ThinkPrune: GRPO with a hard completion-length clip.

    R(y, q; L) = 1[correct(clip(y, L))]

The budget L is enforced upstream as a hard generation cutoff
(``max_completion_length``), so completions reaching the reward function can
never exceed it and no truncation is needed here. Shortening comes entirely
from the clip: an answer that would have needed more than L tokens is cut off
mid-reasoning and scores 0.

Usage:
    accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 \
        train_thinkprune.py --model_name Qwen/Qwen3-8B --max_completion_length 4000
"""
from common import base_arg_parser, run
from data import eval_question, ground_truth, user_question
from transformers import AutoTokenizer

from grading import completion_text, math_score


def parse_args():
    p = base_arg_parser(__doc__.splitlines()[0])
    p.add_argument("--system_prompt", type=str, default=None,
                   help="Overrides the default budget system message. A literal "
                        "'{budget}' is substituted with --max_completion_length.")
    return p.parse_args()


args = parse_args()

SYSTEM_PROMPT = (
    args.system_prompt if args.system_prompt is not None
    else "Think step by step to solve the problem. "
         "The output should be within {budget} tokens."
).replace("{budget}", str(args.max_completion_length))


def thinkprune_reward(completions, solution, **kwargs) -> list[float]:
    return [math_score(completion_text(c), s) for c, s in zip(completions, solution)]


def format_train(example):
    return {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user", "content": user_question(example)}],
        "solution": ground_truth(example),
    }


def format_eval(example):
    return {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user", "content": eval_question(example)}],
        "solution": ground_truth(example),
    }


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    run(args, thinkprune_reward, format_train, format_eval, tokenizer)
