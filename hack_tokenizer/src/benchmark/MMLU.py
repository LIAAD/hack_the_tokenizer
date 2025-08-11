import pandas as pd
from hack_tokenizer.src.benchmark.base import Benchmark
from hack_tokenizer.src.utils import functions
import numpy as np
from typing import Any, Optional, Literal, Union


def benchmark_mmlu(
    benchmark: Benchmark,
    predictions: dict[str, Any]
):
    dataset = benchmark.df.to_dict('records')

    benchmark_output: list[dict[str, Optional[str]]] = []
    for data, n in zip(dataset, range(len(predictions['generated_text']))):
        predicted_text = predictions['generated_text'][n]
        if len(predicted_text) > 1 and not isinstance(predicted_text, str):
            predicted_text = predicted_text[0]

        predicted_answer = functions.get_first_word(data['prompt'], predicted_text)
        benchmark_output.append({
            'prompt': data['prompt'],
            'prediction': predicted_answer,
            'correct_answer': data['answer'],
            'generated_ids': predictions['generated_tokens'][n],
            'generated_logits': predictions.get('logits', {n: 'n/a'})[n]
        })

    accurate_preds = sum(
        str(ben['prediction']).strip().upper() == str(ben['correct_answer']).strip().upper()
        for ben in benchmark_output
    )
    return {
        'benchmark': 'MMLU',
        'accuracy': accurate_preds / len(dataset),
        'accurate_predictions': accurate_preds,
        'wrong_predictions': len(dataset) - accurate_preds,
        'benchmark_predictions': benchmark_output,
    }


class MMLU(Benchmark):

    def __init__(self):
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split="test")

        df = pd.DataFrame({
            'subject': ds['subject'],
            'question': ds['question'],
            'choices': ds['choices'],
            'answer': ds['answer']
        })
        # Convert numeric answer index to letter (A, B, C, D)
        df['answer'] = df['answer'].apply(lambda idx: chr(ord('A') + idx))
        df['prompt'] = df.apply(
            lambda x: f"Question: {x['question']}\nChoices:\n" +
                      "\n".join(f"{chr(ord('A')+i)}. {choice}" for i, choice in enumerate(x['choices'])) +
                      "\nAnswer:",
            axis=1
        )

        output = super().__init__(
            self.__class__.__name__,
            df,
            evaluation_method=benchmark_mmlu,
            prediction_prompts=df['prompt'].tolist(),
            aggregation_method=lambda results: np.array([r['accuracy'] for r in results]).mean()
        )

        self.config['gen_kwargs'] = {
            'num_beams': 1,
            'num_return_sequences': 1,
            'max_new_tokens': 2
        }
        return output

    def get_benchmark_data(self, return_as: Literal['dataframe', 'df', 'list', 'str']='list'):  # type: ignore
        if return_as in ('list', 'str'):
            data: list[str] = self.prediction_prompts.to_list()
            if return_as == 'str':
                return "\n".join(data)
            return data
        return super().get_benchmark_data(return_as=return_as)



if __name__ == '__main__':
    import hack_tokenizer.src.utils.loader as loader
    import torch

    model, tokenizer = loader.load_model_and_tokenizer(
        model_name='HuggingFaceTB/SmolLM3-3B',
        device='cuda',
        model_kwargs={
            'torch_dtype': torch.bfloat16
        },
        tokenizer_kwargs={'padding_side': 'left'}
    )

    mmlu_bench = MMLU()
    generations = mmlu_bench.generate(model, tokenizer)
    results = mmlu_bench.run_eval(model.name_or_path, generations)
    print(results)
