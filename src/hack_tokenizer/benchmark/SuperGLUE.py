from .base import Benchmark
from ..utils import functions
import numpy as np
import pandas as pd
from typing import Any, Callable, Literal, Optional


# ------------------------- Helpers ---------------------------------

def _normalize_binary_prediction(raw: Optional[str], lang: str = "en") -> str:
    """Normalize a raw predicted token/string into '1', '0' or '-1' (unknown).

    Tries several heuristics:
    - If raw is None or empty -> '-1'
    - If first significant character is '0' or '1' -> that value
    - Language-aware startswith checks (yes/no, true/false, sim/não, verdadeiro/falso)
    - Last resort: search for '0' or '1' anywhere in the token and use that
    """
    if raw is None:
        return "-1"
    s = str(raw).strip().lower()
    if s == "":
        return "-1"

    # direct first-char digit
    if s[0] in ("0", "1"):
        return s[0]

    # language-aware checks
    yes_tokens = ("yes", "y", "true", "t", "correct")
    no_tokens = ("no", "n", "false", "f", "incorrect")
    lang = lang.lower()
    if lang == "pt":
        yes_tokens = ("sim", "s", "verdade", "verdadeiro", "v")
        no_tokens = ("não", "nao", "n", "falso", "f")

    for tok in yes_tokens:
        if s.startswith(tok):
            return "1"
    for tok in no_tokens:
        if s.startswith(tok):
            return "0"

    # fallback: try to find a digit 0 or 1 anywhere
    for ch in s:
        if ch in ("0", "1"):
            return ch

    return "-1"

# ------------------------- Task evaluation functions ----------------

def task_boolq(benchmark: Benchmark, predictions: dict[str, Any], lang: str = "en") -> dict:
    df_records = benchmark.df.to_dict("records")
    benchmark_output = []

    for i, (data, prediction) in enumerate(zip(df_records, predictions["generated_text"])):
        # prefer the "first word predicted_answer" helper, fallback to prediction string
        predicted_answer = functions.get_first_word(data.get("prediction_prompts"), prediction)
        if predicted_answer is None:
            predicted_answer = prediction.strip().split()[0] if isinstance(prediction, str) and prediction.strip() else None
        predicted_answer = _normalize_binary_prediction(predicted_answer, lang)
        benchmark_output.append({
            "idx": data.get("idx", data.get("id", i)),
            "input_text": data.get("prediction_prompts", ""),
            "prediction_text": prediction,
            "prediction_label": predicted_answer,
            "correct_label": data.get("label"),
        })

    accurate_preds = sum(1 for r in benchmark_output if str(r["prediction_label"]).strip()[:1] == str(r["correct_label"]))
    return {
        "benchmark": "SuperGLUE: Task BoolQ",
        "accuracy": accurate_preds / len(df_records) if len(df_records) > 0 else 0.0,
        "accurate_predictions": accurate_preds,
        "wrong_predictions": len(df_records) - accurate_preds,
        "benchmark_predictions": benchmark_output,
    }


def task_multirc(benchmark: Benchmark, predictions: dict[str, Any], lang: str = "en") -> dict:
    dataset = benchmark.df.to_dict("records")
    benchmark_output = []

    for i, (data, prediction) in enumerate(zip(dataset, predictions["generated_text"])):
        extracted = functions.get_first_word(data.get("prediction_prompts"), prediction)
        if extracted is None:
            extracted = prediction.strip().split()[0] if isinstance(prediction, str) and prediction.strip() else None
        label = _normalize_binary_prediction(extracted, lang)
        benchmark_output.append({
            "idx": data.get("idx", data.get("id", i)),
            "input_text": data.get("prediction_prompts", ""),
            "prediction_text": prediction,
            "prediction_label": label,
            "correct_label": data.get("label"),
        })

    accurate_preds = sum(1 for r in benchmark_output if str(r["prediction_label"]).strip()[:1] == str(r["correct_label"]))
    return {
        "benchmark": "SuperGLUE: Task MultiRC",
        "accuracy": accurate_preds / len(dataset) if len(dataset) > 0 else 0.0,
        "accurate_predictions": accurate_preds,
        "wrong_predictions": len(dataset) - accurate_preds,
        "benchmark_predictions": benchmark_output,
    }


def task_copa(benchmark: Benchmark, predictions: dict[str, Any], lang: str = "en") -> dict:
    dataset = benchmark.df.to_dict("records")
    benchmark_output = []

    for i, (data, prediction) in enumerate(zip(dataset, predictions["generated_text"])):
        extracted = functions.get_first_word(data.get("prediction_prompts"), prediction)
        if extracted is None:
            extracted = prediction.strip().split()[0] if isinstance(prediction, str) and prediction.strip() else None
        label = _normalize_binary_prediction(extracted, lang=lang)
        benchmark_output.append({
            "idx": data.get("idx", data.get("id", i)),
            "input_text": data.get("prediction_prompts", ""),
            "prediction_text": prediction,
            "prediction_label": label,
            "correct_label": data.get("label"),
        })

    accurate_preds = sum(1 for r in benchmark_output if str(r["prediction_label"]).strip()[:1] == str(r["correct_label"]))
    return {
        "benchmark": "SuperGLUE: Task COPA",
        "accuracy": accurate_preds / len(dataset) if len(dataset) > 0 else 0.0,
        "accurate_predictions": accurate_preds,
        "wrong_predictions": len(dataset) - accurate_preds,
        "benchmark_predictions": benchmark_output,
    }

# ------------------------- SuperGLUE class ---------------------------

class SuperGLUE(Benchmark):
    """Generic SuperGLUE benchmark class.

    Usage:
        SuperGLUE(task='boolq')  # loads HF `super_glue` dataset for the task

    You can pass `df` and/or `prediction_prompts` if you want to fully control
    dataset loading and prompt construction (useful for subclassing).
    """

    TASK_MAPPING = {
        "copa": task_copa,
        "multirc": task_multirc,
        "boolq": task_boolq,
    }
    lang: str='en'

    def __init__(
        self,
        task: Literal["boolq", "multirc", "copa"] = "boolq",
        df: Optional[pd.DataFrame] = None,
        prediction_prompts: Optional[list[str]] = None,
        evaluation_method: Optional[Callable[["Benchmark", dict], Any]] = None,
    ):
        assert task in self.TASK_MAPPING, f"Unknown task: {task}"

        # Load dataset if not provided
        if df is None:
            # `functions.load_dataset_to_dataframe` should load the HF `super_glue` dataset
            # for the chosen task. Adjust arguments if your loader expects another signature.
            df = functions.load_dataset_to_dataframe("super_glue", data_dir=task)

        # Build English prompts if they were not provided
        if prediction_prompts is None:
            if task == "boolq":
                df["prediction_prompts"] = (
                    "Passage: " + df["passage"].fillna("") +
                    "\nQuestion: " + df["question"].fillna("") +
                    "\nAnswer (0-No, 1-Yes):"
                )
            elif task == "multirc":
                df["prediction_prompts"] = (
                    "You will receive a paragraph, a question, and an answer. "
                    "Identify whether the answer is correct (1) or incorrect (0).\n"
                    "Paragraph: " + df["paragraph"].fillna("") +
                    "\nQuestion: " + df["question"].fillna("") +
                    "\nAnswer: " + df["answer"].fillna("") +
                    "\nConclusion (0-Incorrect, 1-Correct):"
                )
            elif task == "copa":
                # Start with a base premise for all rows
                df["prediction_prompts"] = "Premise: " + df["premise"].fillna("") + "\nWhat was "
                # cause vs effect
                df.loc[df["question"] == "cause", "prediction_prompts"] = (
                    df.loc[df["question"] == "cause", "prediction_prompts"]
                    + "the CAUSE of this?\nChoice 0: "
                    + df.loc[df["question"] == "cause", "choice1"].fillna("")
                    + "\nChoice 1: "
                    + df.loc[df["question"] == "cause", "choice2"].fillna("")
                    + "\nAnswer (0 or 1):"
                )
                df.loc[df["question"] == "effect", "prediction_prompts"] = (
                    df.loc[df["question"] == "effect", "prediction_prompts"]
                    + "the EFFECT of this?\nChoice 0: "
                    + df.loc[df["question"] == "effect", "choice1"].fillna("")
                    + "\nChoice 1: "
                    + df.loc[df["question"] == "effect", "choice2"].fillna("")
                    + "\nAnswer (0 or 1):"
                )

        # Resolve evaluation method
        def eval_method(*args, **kwargs):
            if evaluation_method is not None:
                return evaluation_method(*args, **kwargs)
            return self.TASK_MAPPING[task](*args, lang=self.lang, **kwargs)

        # Now initialize the Benchmark base class
        super().__init__(
            name=self.__class__.__name__,
            dataset=df,
            prediction_prompts=prediction_prompts or df["prediction_prompts"].tolist(),
            evaluation_method=eval_method,
            aggregation_method=lambda results: np.array([r["accuracy"] for r in results]).mean(),
        )

