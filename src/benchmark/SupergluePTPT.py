import pandas as pd
import tqdm
from .base import Benchmark, TOKENIZER_TYPE, MODEL_TYPE
import src.utils as utils
import numpy as np


def task_boolq(
    benchmark: Benchmark,
    model: MODEL_TYPE,
    tokenizer: TOKENIZER_TYPE
):

    parallel = benchmark.config['parallel_tasks']
    parallel_group_size = benchmark.config['parallel_batch_size']
    model_kwargs = benchmark.config['model_kwargs']
    MODEL = model.name_or_path
    DEVICE = model.device
    dataset = benchmark.df.to_dict('records')

    benchmark_output: list[dict[str, str]] = []
    tqdm_desc = benchmark.config['tqdm_description'].format(MODEL=MODEL, benchmark_name=benchmark.name)

    # Prepare input texts
    input_texts = []
    for data in dataset:
        passage, question = data['passage'], data['question']
        input_texts.append(f'Passagem: {passage}\nPergunta: {question}\nResposta (0-Verdade, 1-Mentira):')

    # Obtain the predictions using the model generation (either parallel or not)
    predictions = []
    if not parallel:
        # Predict one input_text at a time
        for input_text in tqdm.tqdm(input_texts, desc=tqdm_desc):
            input_tokens = tokenizer.encode(input_text, return_tensors="pt").to(DEVICE)
            predictions.append(model.generate(
                input_tokens,
                max_length=input_tokens.size()[1] + 5,
                **model_kwargs
            ))
    else:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        # Create group of tokens (if there are too many tokens, GPU may not have enough memory)
        groups_of_tokens = [
            tokenizer(input_texts[i*parallel_group_size: (i+1)*parallel_group_size], return_tensors='pt', padding=True, padding_side='left')
            for i in range(len(input_texts) // parallel_group_size)
        ]
        for tokens in tqdm.tqdm(groups_of_tokens, desc=tqdm_desc):
            token_inputs, attention_mask = tokens['input_ids'].to(DEVICE),tokens['attention_mask'].to(DEVICE)
            predictions.extend(model.generate(
                token_inputs,
                attention_mask = attention_mask,
                max_length = token_inputs.shape[1] + 5,  # Generate 5 aditional tokens
                pad_token_id = tokenizer.eos_token_id
            ))
            # Clearing GPU memory
            del token_inputs, attention_mask
    # After obtaining predictions start the evaluation process
    for data, input_text, prediction in zip(dataset, input_texts, predictions):
        if len(prediction.shape) > 1: prediction = prediction[0]
        # Retrieve the actual answer from the prediction
        prediction = tokenizer.decode(prediction, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        predicted_answer = utils.get_first_word(input_text, prediction)
        if not predicted_answer in ['1', '0']:
            if predicted_answer is None:
                predicted_answer = '-1'
            elif predicted_answer.lower().startswith('sim') or predicted_answer.lower().startswith('verda'):
                predicted_answer = '1'
            else:
                predicted_answer = '0'
        benchmark_output.append({
            'idx': data['idx'], 'input_text': input_text, 'prediction_text': prediction, 'prediction_label': predicted_answer, 'correct_label': data['label']
        })
    accurate_preds = sum(ben['prediction_label'].strip()[:1] == str(ben['correct_label']) for ben in benchmark_output)
    return {
        'benchmark': 'Superglue pt-PT: Task BoolQ',
        'accuracy': accurate_preds / len(dataset),
        'accurate_predictions': accurate_preds,
        'wrong_predictions': len(dataset) - accurate_preds,
        'benchmark_predictions': benchmark_output,
        'model': MODEL
    }


class SupergluePTPT(Benchmark):
    def __init__(self):
        # Loading CALAME-PT dataset onto a Pandas DataFrame
        df = utils.load_dataset_to_dataframe('PORTULAN/extraglue', data_dir='data/boolq_pt-PT') 

        return super().__init__(
            self.__class__.__name__,
            df,
            evaluation_method=task_boolq,
            aggregation_method=lambda results: np.array([r['accuracy'] for r in results]).mean()
        )
