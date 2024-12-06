import pandas as pd
import tqdm
from .base import Benchmark, TOKENIZER_TYPE, MODEL_TYPE
import src.utils as utils
import numpy as np


def benchmark(
    benchmark: Benchmark,
    model: MODEL_TYPE,
    tokenizer: TOKENIZER_TYPE
):
    # NOTE: We could improve this benchmark by using the PROBABILITY Distribution of the actual token.
    parallel = benchmark.config['parallel_tasks']
    parallel_group_size = benchmark.config['parallel_batch_size']
    model_kwargs = benchmark.config['model_kwargs']
    MODEL = model.name_or_path
    DEVICE = model.device
    dataset = benchmark.df.to_dict('records')

    benchmark_output: list[dict[str, str]] = []
    tqdm_desc = benchmark.config['tqdm_description'].format(MODEL=MODEL, benchmark_name=benchmark.name)
    
    # Obtain the predictions using the model generation (either parallel or not)
    predictions = []
    if not parallel:
        # Predict one input_text at a time
        for data in tqdm.tqdm(dataset, desc=tqdm_desc):
            input_tokens = tokenizer.encode(data['sentence'], return_tensors="pt").to(DEVICE)
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
            tokenizer([d['sentence'] for d in dataset[i*parallel_group_size: (i+1)*parallel_group_size]], return_tensors='pt', padding=True, padding_side='left')
            for i in range(len(dataset) // parallel_group_size)
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
    for data, prediction in zip(dataset, predictions):
        if len(prediction.shape) > 1: prediction = prediction[0]
        # Retrieve the actual answer from the prediction
        prediction = tokenizer.decode(prediction, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        predicted_answer = utils.get_first_word(data['sentence'], prediction)
        benchmark_output.append({
            'text': data['sentence'], 'prediction': predicted_answer, 'correct_word': data['last_word']
        })
    
    accurate_preds = sum(str(ben['prediction']).lower() == str(ben['correct_word']).lower() for ben in benchmark_output)
    return {
        'benchmark': 'CALAME-PT',
        'accuracy': accurate_preds / len(dataset),
        'accurate_predictions': accurate_preds,
        'wrong_predictions': len(dataset) - accurate_preds,
        'benchmark_predictions': benchmark_output,
        'model': MODEL
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
            aggregation_method=lambda results: np.array([r['accuracy'] for r in results]).mean()
        )
