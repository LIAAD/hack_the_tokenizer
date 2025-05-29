import argparse
import torch
import sys
import pathlib
import tqdm
import os
import re
from typing import Optional
from src import utils, loader, hack
from src.DatasetClass import ListDataset, TextDataset
from torch.utils.data import DataLoader

DEVICE                  = 'cuda'
GENERATION_BATCH_SIZE   = 8
MODEL                   = 'Qwen/Qwen2.5-1.5B-Instruct'
TEMPERATURE             = None

# TODO:
#   1. Comparar ranking de "new_token" sem ter "treino" ou com "treino"
#   2. Nas frases que nao gerou corretamente, comparar ranking de "novo_token" com o primeiro token necessario para gerar a palvra "correta"
#   3. Visualizar um box-plot com os "new_token_rankings".
#   4. Calcular Fertilidade com os novos tokens
#   5. Comparar diferentes tipos de inicializacao de embeddings dos novos tokens (AVG, WeightedAvg, Random, etc)

def main(
    model_name: Optional[str],
    device: Optional[str],
    generation_batch_size: Optional[int],
    temperature: Optional[str]
):
    if temperature is None:
        model_gen_kwargs = dict(top_p=None, top_k=None, temperature=None, do_sample=False)
    else:
        model_gen_kwargs = dict(temperature=temperature)

    # Import model
    model, tokenizer = loader.load_model_and_tokenizer(
        model_name=model_name,
        device=device,
        model_kwargs = { 'torch_dtype': torch.bfloat16},
        tokenizer_kwargs={'padding_side': 'left'}
    )
    original_tokenizer = loader.load_model_and_tokenizer(
        model_name=model_name,
        device='cpu',
        model_kwargs = { 'torch_dtype': torch.bfloat16},
        tokenizer_kwargs={'padding_side': 'left'}
    )[1]



    import src.benchmark as Benchmark
    from src.benchmark.CalamePT import CalamePT

    # Removing "SuperGluePTPT" from the Benchmarks
    benchmark = Benchmark.Benchmarks([CalamePT()])

    # Adding the Batch Size (to generate in parallel)
    benchmark.config['parallel_batch_size'] = GENERATION_BATCH_SIZE
    benchmark.config['max_new_tokens']      = max(len(tokenizer.encode(x)) for x in CalamePT().df['last_word'].values) + 1   # Maximum tokenization of predicted words


    return None

if __name__ == '__main__':
    # ------------------------------------------------------------------------------
    # Parsing arguments from shell (ton execute with "python -m package arg1 arg2 arg3...")
    parser = argparse.ArgumentParser(
        prog='Hack the Tokenizer Evaluation'
    )
    parser.add_argument('-m', '--model', help='Model to run the evaluation for.', default=MODEL)
    parser.add_argument('-d', '--device', help='Device onto which to run the evaluation.', default=DEVICE)
    parser.add_argument('-b', '--batch', help='Batch size for the training section. Specifies how many sentences to process in parallel', default=GENERATION_BATCH_SIZE)
    parser.add_argument('-t', '--temperature', help='Model generation temperature. This allows to manipulate the generation to be Deterministic.', default=TEMPERATURE)

    args = parser.parse_args()
    args['model_name'] = args.pop('model')
    args['generation_batch_size'] = args.pop('batch')

    main(**args)