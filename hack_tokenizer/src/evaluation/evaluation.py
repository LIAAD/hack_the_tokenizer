from typing import Optional 
from pathlib import Path
import datetime as dt
import argparse

import torch
import numpy as np
import transformers

from hack_tokenizer.src.hack import ModelHacker
from .. import loader, benchmark as Benchmark
from ..metrics import METRICS 
from ..json_dumper import dump_json

np.random.seed(42)  # Setting numpy seed to reproduce randomness results

DEVICE                  = 'cuda'
GENERATION_BATCH_SIZE   = 8
# MODEL                   = 'Qwen/Qwen2.5-1.5B-Instruct'
MODEL                   = 'HuggingFaceTB/SmolLM2-135M'
TEMPERATURE             = None
LEARNING_RATE           = 1e-6
NUMBER_NEW_TOKENS       = 1000


# TODO:
#   1. Comparar ranking de "new_token" sem ter "treino" ou com "treino"
#   2. Nas frases que nao gerou corretamente, comparar ranking de "novo_token" com o primeiro token necessario para gerar a palvra "correta"
#   3. Visualizar um box-plot com os "new_token_rankings".
#   4. Calcular Fertilidade com os novos tokens
#   5. Comparar diferentes tipos de inicializacao de embeddings dos novos tokens (AVG, WeightedAvg, Random, etc)

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
        store_generation_data: bool=False,
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
        self.store_generation_data = store_generation_data

    def run_benchmark_and_metrics(self, encode_tokenizer, model_gen_kwargs, store_generation_data ):
       output = {
           'Benchmarks': self.benchmarks.run(self.model, self.tokenizer, encode_tokenizer, model_gen_kwargs, store_generation_data),
           'Metrics': METRICS.run(self.model, self.tokenizer)
       }
       return output 

    def evaluate(self):
        # Setting up configs
        model_gen_kwargs = dict(top_p=None, top_k=None, temperature=None, do_sample=False) if self.temperature is None else dict(temperature=self.temperature)
        if self.dataset_tokenizer is None: self.dataset_tokenizer = Benchmark.BENCHMARKS.get_benchmark_data('list')
        if self.dataset_training is None:  self.dataset_training = self.dataset_tokenizer.copy()
        if self.datasets_metrics is not None and len(self.datasets_metrics) > 0: 
            for metric in self.datasets_metrics.keys():
                METRICS.update_data(self.datasets_metrics, metric)

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
        self.benchmarks.config['parallel_batch_size'] = GENERATION_BATCH_SIZE   # Adding the Batch Size (to generate in parallel)

# ------------------------------------------------------------------------
#                               BASELINE
        # Run benchmark with original model and tokenizer (without any modifications)
        results = {
            'BASELINE': self.run_benchmark_and_metrics(None, model_gen_kwargs, store_generation_data=self.store_generation_data)
        }
# ------------------------------------------------------------------------

# ------------------------------------------------------------------------
#                   BASELINE + Initialization (new tokens)
        # Creating new tokens + adding them to model and initializing them using "weighted_drop(1.5)"
        self.model, self.tokenizer = hacker.hack(
            self.model, self.tokenizer,
            self.encoding_tokenizer,
            num_tokens=self.number_new_tokens,
            embed_initializer_method='weighted_drop(1.5)',
            show_progress=True,
            train=False,
        )
        results['INITIALIZED_NO_TRAINING'] = self.run_benchmark_and_metrics(self.encoding_tokenizer, model_gen_kwargs, self.store_generation_data)
# ------------------------------------------------------------------------    

# ------------------------------------------------------------------------
#                   BASELINE + Initialization & Training
        # Training the model
        self.model = hacker.train(
            self.model, self.tokenizer,
            self.encoding_tokenizer,
            dataset=self.dataset_training
        )
        results['INITIALIZED_WITH_TRAINING'] = self.run_benchmark_and_metrics(self.encoding_tokenizer, model_gen_kwargs, self.store_generation_data)
# ------------------------------------------------------------------------

        # Save results in JSON file
        dump_json(results, f'{self.output_directory}/results_{dt.datetime.now().strftime("%Y%m%d%H%M%S")}.json', False)
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

    Evaluation(**args).evaluate()


if __name__ == '__main__':
    main()