import pandas as pd
from .base import Benchmark
from ..utils import functions
import numpy as np
from typing import Any, Optional


def benchmark(
    benchmark: Benchmark,
    predictions: dict[str, Any]
):
    # NOTE: We could improve this benchmark by using the PROBABILITY Distribution of the actual token.
    dataset = benchmark.df.to_dict('records')

    benchmark_output: list[dict[str, Optional[str]]] = []
    # After obtaining predictions start the evaluation process
    for data, n in zip(dataset, range(len(predictions['generated_text']))):
        predicted_text = predictions['generated_text'][n]
        if len(predicted_text) > 1 and not isinstance(predicted_text, str): predicted_text = predicted_text[0]
        # Retrieve the actual answer from the prediction
        predicted_answer = functions.get_first_word(data['sentence'], predicted_text)
        benchmark_output.append({
            'text': data['sentence'],
            'prediction': predicted_answer,
            'correct_word': data['last_word'],
            'generated_ids': predictions['generated_tokens'][n],
            'generated_logits': predictions.get('logits', {n: 'n/a'})[n]
        })

    
    accurate_preds = sum(str(ben['prediction']).lower().strip() == str(ben['correct_word']).lower().strip() for ben in benchmark_output)
    return {
        'benchmark': 'CALAME-PT',
        'accuracy': accurate_preds / len(dataset),
        'accurate_predictions': accurate_preds,
        'wrong_predictions': len(dataset) - accurate_preds,
        'benchmark_predictions': benchmark_output,
    }


class CalamePT(Benchmark):

    def __init__(self):
        # Loading CALAME-PT dataset onto a Pandas DataFrame
        df_handwritten = pd.read_json("https://huggingface.co/datasets/NOVA-vision-language/calame-pt/resolve/main/calamept_handwritten_only.jsonl", lines=True)
        df_handwritten['Source'] = 'Handwritten'
        df_generated = pd.read_json("https://huggingface.co/datasets/NOVA-vision-language/calame-pt/resolve/main/calamept_gen_only.jsonl", lines=True)
        df_generated['Source'] = 'Generated'
        df = pd.concat([df_handwritten, df_generated])[['id', 'sentence', 'last_word']]

        output = super().__init__(
            self.__class__.__name__,
            df,
            evaluation_method=benchmark,
            prediction_prompts=df['sentence'].unique().tolist(),    # type: ignore (this column is string, so .to_list() returns list[str])
            aggregation_method=lambda results: np.array([r['accuracy'] for r in results]).mean()
        )
        self.config['gen_kwargs'] = {
            'num_beams': 1,
            'num_return_sequences': 1,
        }
        return output 

    def generate(self, model, tokenizer, *args, **kwargs):
        # Before generating, update the config for the "max_new_tokens" to limit it with the MAX encoding of the "last_word"
        #   This way, we guarantee we give possibility for our model to generate the "last_word" tokens
        self.config['gen_kwargs']['max_new_tokens'] = max(len(tokenizer.encode(x)) for x in self.df['last_word'].values) + 1   # Maximum tokenization of predicted words,
        return super().generate(model, tokenizer, *args, **kwargs)


if __name__ == '__main__':
    import hack_tokenizer.src.utils.loader as loader
    import torch
    model, tokenizer = loader.load_model_and_tokenizer(
        model_name='HuggingFaceTB/SmolLM-135M',
        device='cpu',
        model_kwargs={
            'torch_dtype': torch.bfloat16
        },
        tokenizer_kwargs={'padding_side': 'left'}
    )
    calame = CalamePT()
    calame.generate(
        model, tokenizer
    )