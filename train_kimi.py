"""Kimi k1.5 length penalty (arXiv:2501.12599, Sec 2.3.3).

    r = 1[correct(y)] + w * len_reward(y)

    lambda      = 0.5 - (n_y - min_len) / (max_len - min_len)
    len_reward  = lambda            if correct
                  min(0, lambda)    otherwise

min_len and max_len are group-relative: they are computed across the k
rollouts that share a prompt, not globally. A correct-but-short completion is
rewarded, a long one penalised; incorrect completions can only be penalised,
never rewarded for brevity.

There is no length instruction in the prompt -- this reward shapes length
purely through the group comparison.

The penalty can be warmed up (``--length_penalty_warmup_steps``) so the model
first learns to be correct before being pushed to be short.

Usage:
    accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 \
        train_kimi.py --model_name Qwen/Qwen3-8B --length_weight 0.1
"""
from collections import defaultdict

from transformers import AutoTokenizer, TrainerCallback

from common import base_arg_parser, run
from data import eval_question, ground_truth, user_question
from grading import completion_text, math_score


def parse_args():
    p = base_arg_parser(__doc__.splitlines()[0])
    p.add_argument("--length_weight", type=float, default=0.1,
                   help="Weight w on the length reward relative to correctness.")
    p.add_argument("--length_penalty_warmup_steps", type=int, default=100,
                   help="Steps of correctness-only training before the length "
                        "reward is switched on. 0 enables it immediately.")
    return p.parse_args()


args = parse_args()
# Bound at module level because the reward function below reads it as a
# global (see LengthPenaltyState's note on functools.partial).
tokenizer = AutoTokenizer.from_pretrained(args.model_name)


class LengthPenaltyState:
    """Toggled by the warmup callback, read by the reward function.

    Reward functions are read as module-level globals rather than bound with
    functools.partial because GRPOTrainer reads ``reward_func.__name__`` to
    build its reward-name list, and partial objects do not have one.
    """

    def __init__(self, warmup_steps):
        self.warmup_steps = warmup_steps
        self.enabled = warmup_steps == 0


class LengthPenaltyWarmupCallback(TrainerCallback):
    def __init__(self, state):
        self.lp_state = state

    def on_step_end(self, args, state, control, **kwargs):
        if not self.lp_state.enabled and state.global_step >= self.lp_state.warmup_steps:
            self.lp_state.enabled = True
            if state.is_world_process_zero:
                print(f"[length penalty] enabled at step {state.global_step}", flush=True)


length_penalty_state = LengthPenaltyState(args.length_penalty_warmup_steps)


def kimi_length_reward(completions, prompts, solution, **kwargs) -> list[float]:
    texts = [completion_text(c) for c in completions]
    n_y = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]
    correct = [math_score(t, s) for t, s in zip(texts, solution)]

    # Group by prompt content rather than assuming rollouts of the same prompt
    # are contiguous -- GRPOTrainer's batch ordering is not a documented
    # guarantee.
    groups = defaultdict(list)
    for idx, p in enumerate(prompts):
        groups[p[-1]["content"] if isinstance(p, list) else p].append(idx)

    rewards = [0.0] * len(completions)
    for idxs in groups.values():
        lens = [n_y[i] for i in idxs]
        min_len, max_len = min(lens), max(lens)
        for i in idxs:
            if not length_penalty_state.enabled or max_len == min_len:
                length_reward = 0.0
            else:
                lam = 0.5 - (n_y[i] - min_len) / (max_len - min_len)
                length_reward = lam if correct[i] else min(0.0, lam)
            rewards[i] = correct[i] + args.length_weight * length_reward
    return rewards


def format_train(example):
    return {"prompt": [{"role": "user", "content": user_question(example)}],
            "solution": ground_truth(example)}


def format_eval(example):
    return {"prompt": [{"role": "user", "content": eval_question(example)}],
            "solution": ground_truth(example)}


if __name__ == "__main__":
    run(args, kimi_length_reward, format_train, format_eval, tokenizer,
        extra_callbacks=[LengthPenaltyWarmupCallback(length_penalty_state)])
