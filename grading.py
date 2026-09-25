"""Answer extraction and correctness scoring for math problems.

Wraps the vendored Qwen2.5-Math grader (``vendor/qwen_math.py``) in a form
that is safe to call synchronously inside a training process.

``qwen_math.compute_score`` is deliberately not reused: it spawns a
subprocess per comparison, which is both slow and unsafe once CUDA has been
initialised in the parent. Here ``math_equal`` is called directly and the
worst-case SymPy latency is bounded with a SIGALRM timeout instead.
"""
import signal

from vendor.qwen_math import extract_answer, math_equal, strip_string

DEFAULT_TIMEOUT_SECONDS = 5  # signal.alarm() granularity is whole seconds


class _Timeout:
    """Bound a block's wall-clock time with SIGALRM. Main thread only."""

    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def _handle(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._handle)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_value, traceback):
        signal.alarm(0)


def math_score(prediction_text: str, ground_truth: str,
               timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> float:
    """Return 1.0 if the boxed answer in ``prediction_text`` matches
    ``ground_truth``, else 0.0.

    An unparseable or pathological pair scores 0.0 rather than raising, so a
    single bad completion cannot abort training.
    """
    try:
        answer = extract_answer(prediction_text, "math", use_last_number=False)
        if not answer:
            return 0.0
        with _Timeout(seconds=timeout_seconds):
            is_correct = math_equal(answer, strip_string(ground_truth), timeout=False)
        return 1.0 if is_correct else 0.0
    except (TimeoutError, Exception):
        return 0.0


def completion_text(completion) -> str:
    """GRPOTrainer hands back either a raw string or a chat-message list."""
    return completion if isinstance(completion, str) else completion[-1]["content"]
