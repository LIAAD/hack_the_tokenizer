import pandas as pd
from .base import Benchmark
import src.utils as utils
import numpy as np
from typing import Any


def benchmark(
    benchmark: Benchmark,
    predictions: dict[str, Any]
):
    # NOTE: We could improve this benchmark by using the PROBABILITY Distribution of the actual token.
    dataset = benchmark.df.to_dict('records')

    benchmark_output: list[dict[str, str]] = []
    # After obtaining predictions start the evaluation process
    for data, n in zip(dataset, range(len(predictions['generated_text']))):
        predicted_text = predictions['generated_text'][n]
        if len(predicted_text) > 1 and not isinstance(predicted_text, str): predicted_text = predicted_text[0]
        # Retrieve the actual answer from the prediction
        predicted_answer = utils.get_first_word(data['sentence'], predicted_text)
        benchmark_output.append({
            'text': data['sentence'],
            'prediction': predicted_answer,
            'correct_word': data['last_word'],
            'generated_ids': predictions['generated_tokens'][n],
            'generated_logits': predictions.get('logits', {n: 'n/a'})[n]
        })

    
    accurate_preds = sum(str(ben['prediction']).lower() == str(ben['correct_word']).lower() for ben in benchmark_output)
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
        df_handwritten = pd.read_json("hf://datasets/NOVA-vision-language/calame-pt/calamept_handwritten_only.jsonl", lines=True)
        df_handwritten['Source'] = 'Handwritten'
        df_generated = pd.read_json("hf://datasets/NOVA-vision-language/calame-pt/calamept_gen_only.jsonl", lines=True)
        df_generated['Source'] = 'Generated'
        df = pd.concat([df_handwritten, df_generated])[['id', 'sentence', 'last_word']]

        return super().__init__(
            self.__class__.__name__,
            df,
            evaluation_method=benchmark,
            prediction_prompts=df['sentence'].unique().tolist(),
            aggregation_method=lambda results: np.array([r['accuracy'] for r in results]).mean()
        )
