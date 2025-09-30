from typing import Literal

from ..utils import functions
from .SuperGLUE import SuperGLUE



class extraGLUE(SuperGLUE):
    """Portuguese variant of SuperGLUE using the PORTULAN/extraglue dataset.

    This class only overrides dataset loading and prompt wording; it reuses all
    evaluation logic from SuperGLUE.
    """
    lang: str='pt'

    def __init__(self, task: Literal["boolq", "multirc", "copa"] = "boolq"):
        assert task in self.TASK_MAPPING, f"Unknown task: {task}"

        # Load the Portuguese dataset
        df = functions.load_dataset_to_dataframe("PORTULAN/extraglue", data_dir=f"data/{task}_pt-PT")

        # Build Portuguese prompts
        if task == "boolq":
            df["prediction_prompts"] = (
                "Passagem: " + df["passage"].fillna("") +
                "\nPergunta: " + df["question"].fillna("") +
                "\nResposta (0-Não, 1-Sim):"
            )
        elif task == "multirc":
            df["prediction_prompts"] = (
                "Vais receber um parágrafo, uma pergunta e uma resposta. "
                "Identifica se a resposta é correta (1) ou incorreta (0).\n"
                "Parágrafo: " + df["paragraph"].fillna("") +
                "\nPergunta: " + df["question"].fillna("") +
                "\nResposta: " + df["answer"].fillna("") +
                "\nConclusão (0-Incorreta, 1-Correta):"
            )
        elif task == "copa":
            df["prediction_prompts"] = "Premissa: " + df["premise"].fillna("") + "\nQual foi "
            df.loc[df["question"] == "cause", "prediction_prompts"] = (
                df.loc[df["question"] == "cause", "prediction_prompts"]
                + "a CAUSA para isto?\nAlternativa 0: "
                + df.loc[df["question"] == "cause", "choice1"].fillna("")
                + "\nAlternativa 1: "
                + df.loc[df["question"] == "cause", "choice2"].fillna("")
                + "\nEscolha (0) ou (1):"
            )
            df.loc[df["question"] == "effect", "prediction_prompts"] = (
                df.loc[df["question"] == "effect", "prediction_prompts"]
                + "o EFEITO para isto?\nAlternativa 0: "
                + df.loc[df["question"] == "effect", "choice1"].fillna("")
                + "\nAlternativa 1: "
                + df.loc[df["question"] == "effect", "choice2"].fillna("")
                + "\nEscolha (0) ou (1):"
            )

        # Delegate to SuperGLUE constructor with the pre-built df & prompts
        super().__init__(task=task, df=df, prediction_prompts=df["prediction_prompts"].tolist())
