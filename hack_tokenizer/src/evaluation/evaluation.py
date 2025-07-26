from typing import Optional, Literal
from pathlib import Path
import datetime as dt
import argparse

import tqdm
from hack_tokenizer.src.utils import loader
import torch
import numpy as np
import pandas as pd
import transformers

from hack_tokenizer.src.hack import ModelHacker
from hack_tokenizer.src import benchmark as Benchmark
from hack_tokenizer.src.metrics import METRICS 
from hack_tokenizer.src.utils.json_dumper import dump_json
from hack_tokenizer.src.utils.constants import SEED, DEVICE, GENERATION_BATCH_SIZE, MODEL, TEMPERATURE, LEARNING_RATE, NUMBER_NEW_TOKENS 
np.random.seed(SEED)  # Setting numpy seed to reproduce randomness results


# TODO:
# [X]  1. Comparar ranking de "new_token" sem ter "treino" ou com "treino"
# [X]  2. Nas frases que nao gerou corretamente, comparar ranking de "novo_token" com o primeiro token necessario para gerar a palvra "correta"
# [X]  3. Visualizar um box-plot com os "new_token_rankings".
# [X]  4. Calcular Fertilidade com os novos tokens
# [ ]  5. Comparar diferentes tipos de inicializacao de embeddings dos novos tokens (AVG, WeightedAvg, Random, etc)

class Evaluation:
    def __init__(
        self,
        model_name: str,
        device: str,
        generation_batch_size: int,
        temperature: Optional[str],
        learning_rate: float,
        number_new_tokens: int,
        dataset_tokenizer: Optional[list[str]],
        dataset_training: Optional[list[str]],
        datasets_metrics: Optional[dict[str, list[str]]],
        output_directory: str,
        output_format: Literal['parquet', 'feather', 'csv', 'xlsx'],
        store_generation_data: bool=False,
        embed_init_method: str | Literal['min', 'mean', 'avg', 'quantile({number})', 'weighted_drop({number})'] = 'weighted_drop(1.5)',
        **_
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.generation_batch_size = generation_batch_size
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.number_new_tokens = number_new_tokens
        self.dataset_tokenizer = dataset_tokenizer
        self.dataset_training = dataset_training
        self.datasets_metrics = datasets_metrics
        self.output_directory = output_directory
        self.output_format = output_format
        self.store_generation_data = store_generation_data
        self.embed_init_method = embed_init_method
        self.config = {
            'model_name': model_name,
            'device': device,
            'generation_batch_size': generation_batch_size,
            'temperature': temperature,
            'learning_rate': learning_rate,
            'number_new_tokens': number_new_tokens,
            'output_directory': output_directory,
            'output_format': 'output_format',
            'store_generation_data': store_generation_data,
        }

    def run_benchmark_and_metrics(self, encode_tokenizer, model_gen_kwargs, store_generation_data ):
       output = {
           'Benchmarks': self.benchmarks.run(self.model, self.tokenizer, encode_tokenizer, model_gen_kwargs, store_generation_data),
           'Metrics': METRICS.run(self.model, self.tokenizer)
       }
       return output 

    def run_analysis(self, encoding_tokenizer, tokens_to_analyze: list[str]=[], dataset: list[str]=[], model_gen_kwargs: dict={}):
        # TODO: 1. Add rank and logit of the token necessary to generate the correct word
        #       2. Add the logit of the chosen token
        #       3. Add the logit of the new tokens
        # This should be executed using the `dataset`:
        #   For all "token_tracker" in the dataset, use phrases which should generate any token in the "token_tracker" and predict the ranking and logits of said tokens being chosen
        results = []
        for new_token in tqdm.tqdm(tokens_to_analyze, desc='Creating analaysis for model `{}`'.format(self.model.name_or_path)):
            new_token_id = self.tokenizer.encode(new_token)[0]
            old_token = encoding_tokenizer.tokenize(new_token)[0]
            old_token_id = encoding_tokenizer.encode(old_token)[0]
            phrases_to_generate_new_token = [p for phrase in dataset for p in phrase.split(new_token)[:-1] if new_token in phrase and len(p) > 0]
            for phrase in tqdm.tqdm(phrases_to_generate_new_token,  desc=f'  Generating for new_token=`{new_token}` ', leave=False):
                tokenization = self.tokenizer(phrase, return_tensors='pt')
                input_ids, attention_mask = tokenization['input_ids'].to(self.model.device), tokenization['attention_mask'].to(self.model.device) # type: ignore (all torch.Tensors)
                gen = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_logits=True,
                    pad_token_id=encoding_tokenizer.eos_token_id,
                    **model_gen_kwargs
                )
                new_token_logits = gen.logits[0][0, new_token_id].item()  # type: ignore (gen is a torch.Tensor)
                new_token_rank = (gen.logits[0] > new_token_logits).sum().item()    # type: ignore (gen is a torch.Tensor)
                old_token_logits = gen.logits[0][0, old_token_id].item()  # type: ignore (gen is a torch.Tensor)
                old_token_rank = (gen.logits[0] > old_token_logits).sum().item()    # type: ignore (gen is a torch.Tensor)
                results.append({
                    'phrase': phrase,
                    'new_token': new_token,
                    'new_token_id': new_token_id,
                    'new_token_rank': new_token_rank,
                    'new_token_logits': new_token_logits,
                    'old_token': old_token,                 # First token required to generate the word E.g: "chegada" = "che"+"gada", would be the "che" token
                    'old_token_id': old_token_id,           # Id for the "old_token"
                    'old_token_logits': old_token_logits,   # Logits for the first token required to generated the expected word
                    'old_token_rank': old_token_rank,        # 
                })
        return results        

    def evaluate(self):
        # Setting up configs
        model_gen_kwargs = dict(top_p=None, top_k=None, temperature=None, do_sample=False) if self.temperature is None else dict(temperature=self.temperature)
        if self.dataset_tokenizer is None: self.dataset_tokenizer = Benchmark.BENCHMARKS.get_benchmark_data('list')
        if self.dataset_training is None:  self.dataset_training = self.dataset_tokenizer.copy()
        if self.datasets_metrics is not None and len(self.datasets_metrics) > 0: 
            for metric in self.datasets_metrics.keys():
                METRICS.update_data(self.datasets_metrics[metric], metric)

        # Import model
        self.model, self.tokenizer = loader.load_model_and_tokenizer(
            model_name=self.model_name,
            device=self.device,
            model_kwargs = {'torch_dtype': torch.bfloat16},
            tokenizer_kwargs={'padding_side': 'left'}
        )
        # Save copy of original tokenizer to keep it for encoding before sending to model
        self.encoding_tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
        hacker = ModelHacker(
            dataset=self.dataset_tokenizer,
            batch_size=self.generation_batch_size,
            learning_rate=self.learning_rate
        )

        # Removing "SuperGluePTPT" from the Benchmarks
        self.benchmarks = Benchmark.BENCHMARKS
        self.benchmarks.config['parallel_batch_size'] = self.generation_batch_size   # Adding the Batch Size (to generate in parallel)

# ------------------------------------------------------------------------
#                               BASELINE
        # Run benchmark with original model and tokenizer (without any modifications)
        results = {
            'BASELINE': self.run_benchmark_and_metrics(None, model_gen_kwargs, store_generation_data=self.store_generation_data)
        }
# ------------------------------------------------------------------------

# ------------------------------------------------------------------------
#                   BASELINE + Initialization (new tokens)
        # Creating new tokens + adding them to model and initializing them using "embed_init_method"
        self.model, self.tokenizer = hacker.hack(
            self.model, self.tokenizer,
            self.encoding_tokenizer,
            num_tokens=self.number_new_tokens,
            embed_initializer_method=self.embed_init_method,
            show_progress=True,
            train=False,
        )
        self.model.name_or_path = f'{self.model.name_or_path}[NEW_TOKENS]'
        results['INITIALIZED_NO_TRAINING'] = self.run_benchmark_and_metrics(self.encoding_tokenizer, model_gen_kwargs, self.store_generation_data)
        df = pd.DataFrame(self.run_analysis(self.encoding_tokenizer, hacker.new_tokens, dataset=self.dataset_training, model_gen_kwargs=model_gen_kwargs))
        df['model'] = self.model.name_or_path
# ------------------------------------------------------------------------    

# ------------------------------------------------------------------------
#                   BASELINE + Initialization & Training
        # Training the model
        self.model = hacker.train(
            self.model, self.tokenizer,
            self.encoding_tokenizer,
            dataset=self.dataset_training
        ) 
        self.model.name_or_path = self.model.name_or_path.replace('[NEW_TOKENS]', '[NEW_TOKENS_TRAINED]')
        results['INITIALIZED_WITH_TRAINING'] = self.run_benchmark_and_metrics(self.encoding_tokenizer, model_gen_kwargs, self.store_generation_data)
        df2 = pd.DataFrame(self.run_analysis(self.encoding_tokenizer, hacker.new_tokens, dataset=self.dataset_training, model_gen_kwargs=model_gen_kwargs)) 
        df2['model'] = self.model.name_or_path
        df = pd.concat([df, df2])
# ------------------------------------------------------------------------

        # Add CONFIG to the "results" json (storing the configuration to output)
        results = {
            'RUN_CONFIGS': self.config,
            'RESULTS': results
        }
        # Save results in JSON file
        version = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        dump_json(results, f'{self.output_directory}/results_{version}.json', False)
        
        # Save analysis results on the specified format
        output_path = f'{self.output_directory}/analysis_{version}.{self.output_format}'
        match self.output_format:
            case 'parquet':
                loader.optimize_dataframe(df).to_parquet(output_path)
            case 'feather':
                loader.optimize_dataframe(df).to_feather(output_path)
            case 'csv':
                df.to_csv(output_path, index=False)
            case 'xlsx':
                df.to_excel(output_path, index=False)
        return None


def parse_path_dict(string):
    if not string:  # Handle empty string case
        return {}
    pairs = string.split(',')
    return {k: Path(v) if v else '' for k, v in (pair.split('=') for pair in pairs)}



def main():
    # ------------------------------------------------------------------------------
    # Parsing arguments from shell (ton execute with "python -m package arg1 arg2 arg3...")
    parser = argparse.ArgumentParser(
        prog='Hack the Tokenizer Evaluation'
    )
    parser.add_argument('-mo',     '--model',               type=str,    default=MODEL,                  help='Model to run the evaluation for.')
    parser.add_argument('-d',      '--device',              type=str,    default=DEVICE,                 help='Device onto which to run the evaluation.')
    parser.add_argument('-b',      '--batch',               type=int,    default=GENERATION_BATCH_SIZE,  help='Batch size for the training section. Specifies how many sentences to process in parallel')
    parser.add_argument('-t',      '--temperature',         type=float,  default=TEMPERATURE,            help='Model generation temperature. This allows to manipulate the generation to be Deterministic.')
    parser.add_argument('-l',      '--learning_rate',       type=float,  default=LEARNING_RATE,          help='Learning rate used for the embedding training section.')
    parser.add_argument('-n',      '--number_new_tokens',   type=int,    default=NUMBER_NEW_TOKENS,      help='Number of new tokens to add to the original model.')
    parser.add_argument('-dt',     '--dataset_tokenizer',   type=str,    default=None,                   help='Path to a `TXT` file containing training data for the BytePairEncoding algorithm for the `new_tokens`. (decoded using UTF-8).')
    parser.add_argument('-dT',     '--dataset_training',    type=str,    default=None,                   help='Path to a `TXT` file containing training data. (decoded using UTF-8).')
    parser.add_argument('-out',    '--output_directory',    type=str,    default='./outputs',            help='Directory where to save the json results')
    parser.add_argument('-oF',     '--output_format',       type=str,    default='parquet',              help='Format of the analysis output. Defaults to `parquet` to reduce space usage, but can be one of `parquet`, `csv`, `xlsx`.')
    parser.add_argument('-in_met', '--embed_init_method',   type=str,    default='weighted_drop(1.5)',   help='Specifies Embeddings Initialization Method to use for the "new_tokens". Methods allowed: ["min" "mean" "mean" "avg" "quantile({number})" "weighted_drop({number})')
    parser.add_argument('-dM',     '--datasets_metrics',    type=parse_path_dict, default='',   help='''Mapping of metrics names to their corresponding data paths. 
Format: metric1=/path/to/data1,metric2=/path/to/data2
Example: --dataset_metrics Perplexity=/data/train.csv,Fertility=/data/test.csv
Each key represents a dataset name and its value should be the path to that dataset's data.''')

    args = dict(parser.parse_args()._get_kwargs())
    args['model_name'] = args.pop('model')
    args['generation_batch_size'] = args.pop('batch')

    # Reading `dataset` files (if they exist)
    for key in ['dataset_tokenizer', 'dataset_training']:
        if args.get(key) is not None:
            with open(args[key], 'r', encoding='utf-8') as f:
                args[key] = f.readlines()
    key = 'datasets_metrics'
    for metric in args.get(key, {}):
        with open(args[key][metric], 'r', encoding='utf-8') as f:
            args[key][metric] = f.readlines()


    Evaluation(**args).evaluate()


if __name__ == '__main__':
    main()