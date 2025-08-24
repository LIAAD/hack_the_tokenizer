import argparse
from typing import Optional

import torch
import transformers

from .ModelHacker import ModelHacker
from ..utils import loader


# Colors (ANSI escape sequences)
GREEN  = "\033[0;32m"
WHITE  = "\033[0;37m"
YELLOW = "\033[0;33m"
RESET  = "\033[0m"  # Resets color to default


def parse_args():
    # ------------------------------------------------------------------------------
    # Parsing arguments from shell (ton execute with "python -m package arg1 arg2 arg3...")
    parser = argparse.ArgumentParser(
        prog='Hack the Tokenizer Evaluation'
    )
    parser.add_argument('-mo',     '--model',               default=None,   type=str,    help='Model to run the evaluation for.')
    parser.add_argument('-d',      '--device',              default=None,   type=str,    help='Device onto which to run the evaluation.')
    parser.add_argument('-b',      '--batch',               default=None,   type=int,    help='Batch size for the training section. Specifies how many sentences to process in parallel')
    parser.add_argument('-t',      '--temperature',         default=None,   type=float,  help='Model generation temperature. This allows to manipulate the generation to be Deterministic.')
    parser.add_argument('-l',      '--learning_rate',       default=None,   type=float,  help='Learning rate used for the embedding training section.')
    parser.add_argument('-n',      '--number_new_tokens',   default=None,   type=int,    help='Number of new tokens to add to the original model.')
    parser.add_argument('-dt',     '--dataset_tokenizer',   default=None,   type=str,    help='Path to a `TXT` file containing training data for the BytePairEncoding algorithm for the `new_tokens`. (decoded using UTF-8).')
    parser.add_argument('-dT',     '--dataset_training',    default=None,   type=str,    help='Path to a `TXT` file containing training data. (decoded using UTF-8).')
    parser.add_argument('-in_met', '--embed_init_method',   default=None,   type=str,    help='Specifies Embeddings Initialization Method to use for the "new_tokens". Methods allowed: ["min" "mean" "mean" "avg" "quantile({number})" "weighted_drop({number})')
    parser.add_argument('-nT',     '--max_new_tokens',      default=20,     type=str,    help='Specifies maximum tokens the model is allowed to generate at a time')
    parser.add_argument('-sW',     '--stop_words', default='<end>,</end>',  type=str,    help='List of words which stop generation of model. Should be separated by commas')
    parser.add_argument('-hM',     '--hack_model',         default='True',  type=str,    help='Whether to hack the model or not')

    args = dict(parser.parse_args()._get_kwargs())

    # Reading `dataset` files (if they exist)
    for key in ['dataset_tokenizer', 'dataset_training']:
        if args.get(key) is not None:
            with open(args[key], 'r', encoding='utf-8') as f:
                args[key] = f.readlines()

    args['hack_model'] = args['hack_model'].lower() == 'true'
    return args



def prompt_loop(model, tokenizer, encoding_tokenizer, max_new_tokens: int, stop_words: list[str], temperature: Optional[float]):
    prompts = []
    print(f"\n\n{'':-^100s}{YELLOW}\n\n{'You''re now in prompt mode.':^100s}\n{'Type `quit` or `q` to exit this mode.':^100s}{RESET}\n\n{'':-^100s}")
    
    while True:
        prompt = input(f'\n{RESET}Prompt: ')
        if prompt.lower().startswith('/q') or prompt.lower() in ['q', 'quit', 'exit'] or prompt.lower().startswith(r'\q'):
            return True
        prompts.append({'mode': 'user', 'content': prompt})
    
        prepared_content = '\n'.join([f'<{prompt["mode"]}>{prompt["content"]}</{prompt["mode"]}>' for prompt in prompts])
        prepared_content = f'{prepared_content}\nAssistant: '
        print(f'{RESET}{GREEN}Assistant: ', end='')
        prompts.append({'mode': 'Assistant', 'content': ModelHacker.prompt(model, tokenizer, encoding_tokenizer, prepared_content, max_new_tokens, stop_words, temperature)})
    

def main():
    # Parse shell arguments
    args = parse_args()

    # Import model
    model, tokenizer = loader.load_model_and_tokenizer(
        model_name=args['model'],
        device=args['device'],
        model_kwargs = {'torch_dtype': torch.bfloat16},
        tokenizer_kwargs={'padding_side': 'left'}
    )
    # Save copy of original tokenizer to keep it for encoding before sending to model
    encoding_tokenizer = transformers.AutoTokenizer.from_pretrained(args['model'], padding_side='left')

    if args.get('hack_model'):
        # Hack the model
        hacker = ModelHacker(
            dataset=args['dataset_tokenizer'],
            batch_size=args['batch'],
            learning_rate=args['learning_rate']
        )
        hacker.hack(
            model, tokenizer,
            encoding_tokenizer,
            num_tokens=args['number_new_tokens'],
            embed_initializer_method=args['embed_init_method'],
            show_progress=True,
            train=False if args['dataset_training'] is None else True,
            train_kwargs={
                'dataset': args['dataset_training'],
                'batch_size': args['batch']
            }
        )
    
    # Start prompting
    prompt_loop(model, tokenizer, encoding_tokenizer, int(args['max_new_tokens']), args['stop_words'].split(','), args['temperature'])
    return True

if __name__ == '__main__':
    main()
