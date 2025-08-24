from .base import Benchmark
from ..utils import functions
import numpy as np
from typing import Any


def task_boolq(
    benchmark: Benchmark,
    predictions: dict[str, Any]
):
    dataset = benchmark.df.to_dict('records')

    benchmark_output: list[dict[str, str]] = []
    # After obtaining predictions start the evaluation process
    for data, prediction in zip(dataset, predictions['generated_text']):
        # if len(prediction) > 1: prediction = prediction[0]
        # Retrieve the actual answer from the prediction
        predicted_answer = functions.get_first_word(data['prediction_prompts'], prediction)
        if not predicted_answer in ['1', '0']:
            if predicted_answer is None:
                predicted_answer = '-1'
            elif predicted_answer.lower().startswith('sim') or predicted_answer.lower().startswith('verda'):
                predicted_answer = '1'
            elif predicted_answer.lower().startswith('não') or predicted_answer.lower().startswith('mentir'):
                predicted_answer = '0'
        benchmark_output.append({
            'idx': data['idx'], 'input_text': data['prediction_prompts'], 'prediction_text': prediction, 'prediction_label': predicted_answer, 'correct_label': data['label']
        })
    accurate_preds = sum(ben['prediction_label'].strip()[:1] == str(ben['correct_label']) for ben in benchmark_output)
    return {
        'benchmark': 'Superglue pt-PT: Task BoolQ',
        'accuracy': accurate_preds / len(dataset),
        'accurate_predictions': accurate_preds,
        'wrong_predictions': len(dataset) - accurate_preds,
        'benchmark_predictions': benchmark_output,
    }


class SupergluePTPT(Benchmark):
    def __init__(self):
        # Loading CALAME-PT dataset onto a Pandas DataFrame
        df = functions.load_dataset_to_dataframe('PORTULAN/extraglue', data_dir='data/boolq_pt-PT') 
        # Prepare input texts
        df['prediction_prompts'] = 'Passagem: ' + df['passage'] + '\nPergunta: ' + df['question'] + '\nResposta (0-Verdade, 1-Mentira):'

        return super().__init__(
            self.__class__.__name__,
            df,
            evaluation_method=task_boolq,
            prediction_prompts=df['prediction_prompts'].tolist(),
            aggregation_method=lambda results: np.array([r['accuracy'] for r in results]).mean()
        )
