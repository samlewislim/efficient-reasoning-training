"""Training and evaluation datasets.

Training data is the ``numina_amc_aime`` subset of PRIME-RL/Eurus-2-RL-Data.
Evaluation is AIME, from the parquet bundled with the ThinkPrune release
(vendored under ``aux_data/`` since no equivalent Hub dataset exists).

Each method formats its prompts differently, so the formatters are passed in
rather than fixed here:

  ThinkPrune   system message stating the token budget
  Kimi         no length instruction at all
  LCPO-Exact   "Think for {n} tokens." appended to the user turn
"""
import os

from datasets import load_dataset
from huggingface_hub import snapshot_download

TRAIN_REPO = "PRIME-RL/Eurus-2-RL-Data"
DATA_SOURCE = "numina_amc_aime"
AIME_EVAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "aux_data", "test_aime.parquet")

# Eurus rows carry a fixed PRIME "action" system prompt specific to their own
# pipeline. It is dropped; the question text already ends with an instruction
# to box the final answer, which is what the grader looks for.
def user_question(example) -> str:
    return next(m["content"] for m in example["prompt"] if m["role"] == "user")


def _load_train(split: str = "train"):
    path = snapshot_download(TRAIN_REPO, repo_type="dataset",
                             token=os.environ.get("HF_TOKEN"),
                             endpoint=os.environ.get("HF_ENDPOINT"))
    ds = load_dataset(path, split=split)
    return ds.filter(lambda x: x["data_source"] == DATA_SOURCE)


def build_train_dataset(format_fn, samples=None, seed=3407):
    """``format_fn(example) -> dict`` with at least ``prompt`` and ``solution``."""
    ds = _load_train()
    ds = ds.map(format_fn, remove_columns=ds.column_names)
    if samples is not None and samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(samples))
    return ds


def build_eval_dataset(format_fn, eval_source="AIME22", eval_samples=None):
    ds = load_dataset("parquet", data_files=AIME_EVAL_PATH, split="train")
    ds = ds.filter(lambda x: x["data_source"] == eval_source)
    if eval_samples is not None and eval_samples < len(ds):
        ds = ds.select(range(eval_samples))
    return ds.map(format_fn, remove_columns=ds.column_names)


# Raw AIME problems have no boxed-answer instruction baked in, unlike Eurus
# rows. Append the same one so the grader has something to extract.
BOXED_SUFFIX = "\n\nPresent the answer in LaTeX format: \\boxed{Your answer}"


def eval_question(example) -> str:
    return example["prompt"][0]["content"] + BOXED_SUFFIX


def ground_truth(example) -> str:
    return example["reward_model"]["ground_truth"]
