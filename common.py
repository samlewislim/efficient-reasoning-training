"""Shared GRPO training setup for all three length-control methods.

Everything that does not depend on the reward function lives here: the CLI
surface, LoRA configuration, GRPOConfig construction, and the two callbacks.
Each ``train_*.py`` script supplies only its reward function, its prompt
formatters, and any method-specific arguments.
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from peft import LoraConfig
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from data import build_eval_dataset, build_train_dataset

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
IS_MAIN_PROCESS = int(os.environ.get("RANK", "0")) == 0

# Seven dense projections rather than "all-linear", so the trained-parameter
# count is identical across methods and model families.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]


def base_arg_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Thinking-2507")
    p.add_argument("--samples", type=int, default=None,
                   help="Cap on training rows after filtering.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seeds batch ordering; vary it across runs that share a dataset.")
    p.add_argument("--num_generations", type=int, default=8,
                   help="Rollouts per prompt (GRPO group size).")
    p.add_argument("--num_generations_eval", type=int, default=2,
                   help="Rollouts per prompt at eval. Kept low and independent of "
                        "--num_generations: eval computes entropy over the whole batch "
                        "in one allocation, which can exhaust memory at training's group size.")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=64,
                   help="pdbs * this * world_size / num_generations = prompts per optimizer step.")
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.0,
                   help="KL coefficient against the reference model. 0.0 disables KL "
                        "regularisation and skips loading a reference model.")
    p.add_argument("--max_steps", type=int, default=-1,
                   help="If > 0, overrides --num_train_epochs.")
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--max_completion_length", type=int, default=4000,
                   help="Hard generation cap. For ThinkPrune this is the budget L itself; "
                        "for the other methods it is only a safety ceiling.")
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3,
                   help="Colocate mode keeps a separate vLLM weight copy alongside the "
                        "training copy on the same GPU, so this competes with the trained model.")
    p.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    p.add_argument("--vllm_enable_sleep_mode", action="store_true",
                   help="Release vLLM's memory during the backward pass and reclaim it "
                        "before the next rollout phase.")
    p.add_argument("--vllm_importance_sampling_mode", type=str, default="token_truncate",
                   choices=["token_truncate", "token_mask", "sequence_truncate", "sequence_mask"],
                   help="Correction for the vLLM-generation vs training-forward logprob "
                        "mismatch. Token-level modes are used here because the sequence-level "
                        "default compounds a small per-token mismatch multiplicatively over "
                        "long completions, driving whole sequences out of the loss.")
    p.add_argument("--report_to", type=str, default="wandb")
    p.add_argument("--eval_source", type=str, default="AIME22")
    p.add_argument("--eval_samples", type=int, default=None)
    p.add_argument("--eval_steps", type=int, default=50,
                   help="Also used as save_steps.")
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                   help="checkpoint-N directory from a previous run of the same config.")
    return p


class StepTimingCallback(TrainerCallback):
    """Per-step wall-clock and peak memory, with a projected total.

    Peak memory is max-reduced across ranks: completions are budget-capped
    rather than fixed-length, so different ranks see different pressure in a
    given step and the worst-case GPU is the useful number.
    """

    def __init__(self):
        self.total_steps = None
        self._t0 = None
        self._durations = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.total_steps = state.max_steps

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        self._t0 = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        dt = time.time() - self._t0
        self._durations.append(dt)

        peak_bytes = torch.cuda.max_memory_allocated()
        if dist.is_available() and dist.is_initialized():
            # Must run on every rank: gating the collective on rank 0 hangs.
            t = torch.tensor([peak_bytes], device="cuda", dtype=torch.float64)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            peak_bytes = t.item()

        if not state.is_world_process_zero:
            return
        steady = self._durations[1:] or self._durations  # step 1 includes warmup
        avg = sum(steady) / len(steady)
        msg = (f"[step {state.global_step}/{self.total_steps}] {dt:.1f}s "
               f"(avg {avg:.1f}s/step, peak {peak_bytes / 1e9:.1f} GB worst-GPU)")
        if self.total_steps:
            msg += f" -- projected total: {avg * self.total_steps / 3600:.2f}h"
        print(msg, flush=True)


class BestCheckpointTracker(TrainerCallback):
    """Track the best checkpoint by eval reward.

    ``load_best_model_at_end`` with ``metric_for_best_model="reward"`` does
    not work with GRPOTrainer: it injects ``eval_reward`` into a new local
    dict before calling ``super().log()``, so the metrics dict that
    ``Trainer.evaluate()`` returns never contains it. ``on_log`` does receive
    the enriched dict, so the best step is tracked from there instead.
    """

    def __init__(self):
        self.best_reward = float("-inf")
        self.best_step = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "eval_reward" not in logs:
            return
        is_best = logs["eval_reward"] > self.best_reward
        if is_best:
            self.best_reward = logs["eval_reward"]
            self.best_step = state.global_step
        if state.is_world_process_zero:
            print(f"[eval] step {state.global_step}: eval_reward={logs['eval_reward']:.4f}"
                  f"{' (new best)' if is_best else ''} -- best so far: "
                  f"{self.best_reward:.4f} at step {self.best_step}", flush=True)


def prompt_token_length(tokenizer, prompt) -> int:
    """Render the chat template to text, then tokenise explicitly.

    ``apply_chat_template(tokenize=True)`` has returned different types across
    transformers versions; taking len() of the wrong one silently yields a
    tiny number rather than raising.
    """
    text = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def fit_prompt_lengths(tokenizer, train_dataset, eval_dataset):
    """Drop the longest 10% of training prompts and return the length bound.

    The bound is widened, never shrunk, to fit the longest eval prompt, so
    vLLM's KV cache is sized for what is actually generated rather than the
    model's full declared context.
    """
    lengths = [prompt_token_length(tokenizer, r["prompt"]) for r in train_dataset]
    assert min(lengths) > 10, "implausibly short prompt -- check the chat template"
    max_prompt_length = int(np.quantile(lengths, 0.9)) + 1
    train_dataset = train_dataset.select(
        np.where(np.array(lengths) <= max_prompt_length)[0])

    eval_lengths = [prompt_token_length(tokenizer, r["prompt"]) for r in eval_dataset]
    max_prompt_length = max(max_prompt_length, max(eval_lengths) + 1)
    return train_dataset, max_prompt_length


def build_trainer(args, reward_func, train_dataset, eval_dataset, tokenizer,
                  max_prompt_length, extra_callbacks=()):
    # TRL requires (per_device_eval_batch_size * world_size) to be a multiple
    # of num_generations_eval -- note eval, not training, group size.
    eval_batch_size = args.num_generations_eval // math.gcd(
        args.num_generations_eval, WORLD_SIZE)

    config = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        num_generations_eval=args.num_generations_eval,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=eval_batch_size,
        temperature=0.8,
        top_p=0.95,
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        seed=args.seed,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Explicit load dtype: bf16=True only controls autocast during
        # training, not the dtype the weights are materialised in.
        model_init_kwargs={"dtype": "bfloat16"},
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=max_prompt_length + args.max_completion_length + 64,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_enable_sleep_mode=args.vllm_enable_sleep_mode,
        vllm_importance_sampling_mode=args.vllm_importance_sampling_mode,
        report_to=args.report_to,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=60,
    )

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )

    best_tracker = BestCheckpointTracker()
    trainer = GRPOTrainer(
        model=args.model_name,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        callbacks=[StepTimingCallback(), best_tracker, *extra_callbacks],
    )
    return trainer, best_tracker


def run(args, reward_func, train_format_fn, eval_format_fn, tokenizer,
        extra_callbacks=()):
    """Shared entry point: build data, train, save, report the best step.

    ``tokenizer`` is supplied by the caller rather than created here: the Kimi
    and LCPO reward functions need it as a module-level global, and it must
    already be bound before training starts.
    """
    train_dataset = build_train_dataset(train_format_fn, samples=args.samples)
    eval_dataset = build_eval_dataset(eval_format_fn, args.eval_source, args.eval_samples)

    train_dataset, max_prompt_length = fit_prompt_lengths(
        tokenizer, train_dataset, eval_dataset)

    if IS_MAIN_PROCESS:
        prompts_per_step = (args.per_device_train_batch_size
                            * args.gradient_accumulation_steps
                            * WORLD_SIZE) // args.num_generations
        print(f"world size: {WORLD_SIZE}")
        print(f"prompts per optimizer step: {prompts_per_step}")
        print(f"train rows: {len(train_dataset)}  eval rows: {len(eval_dataset)}")
        print(f"max prompt length: {max_prompt_length}")

    trainer, best_tracker = build_trainer(
        args, reward_func, train_dataset, eval_dataset, tokenizer,
        max_prompt_length, extra_callbacks)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if trainer.accelerator.is_main_process and best_tracker.best_step is not None:
        print(f"Best eval_reward {best_tracker.best_reward:.4f} at step "
              f"{best_tracker.best_step} -- see {args.output_dir}/"
              f"checkpoint-{best_tracker.best_step}")

    # This saves the LAST step, not the best one; see the message above.
    trainer.save_model(f"{args.output_dir}/final")
    return trainer
