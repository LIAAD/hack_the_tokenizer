import re
import os
import json
import torch
import shutil
import logging
import tempfile
import pandas as pd
from collections import OrderedDict

from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer


logger = logging.getLogger(__name__)


def compute_perplexity(model, tokenizer, text: str | list[str]) -> float:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    loss = model(
        input_ids=inputs["input_ids"], labels=inputs["input_ids"], attention_mask=inputs["attention_mask"]
    ).loss
    return torch.exp(loss).item()


def batch_iterator(lines, batch_size: int = 8):
    for i in range(0, len(lines), batch_size):
        yield lines[i : i + batch_size]


def train_tokenizer(tokenizer, dataset, vocab_size: int = 1_000):
    dataset_iterator = batch_iterator(dataset)
    trained_tokenizer = tokenizer.train_new_from_iterator(dataset_iterator, vocab_size=vocab_size)
    return trained_tokenizer


def add_token(token, tokenizer, model):
    tokenizer.add_tokens(token)
    model.resize_token_embeddings(len(tokenizer))


def add_token_to_tokenizer(tokenizer_path: Path, new_token: str) -> None:
    """A very hacky way to add a token to a tokenizer.

    This function adds the required tokens to the tokenizer's vocab.json, merges.txt, and tokenizer.json files of a Huggingface transformer tokenizer.
    """
    logger.info(f"Adding token {new_token} to the tokenizer.")
    if new_token.replace("Ġ", " ").strip() == "":
        logger.info("Empty token, skipping.")
        return

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    tokens, merges = [], []
    steps = tokenizer.tokenize(new_token.replace("Ġ", " "))
    if len(steps) == 1:
        logger.info(f"Token {new_token} is already in the tokenizer.")
        return

    for idx in range(1, len(steps)):
        tkn1 = r"".join(steps[:idx])
        tkn2 = steps[idx]
        merges.append(f"{tkn1} {tkn2}")
        tokens.append(f"{tkn1}{tkn2}")

    original_merges = (tokenizer_path / "merges.txt").read_text().strip().split("\n")
    merges = original_merges + merges
    (tokenizer_path / "merges.txt").write_text("\n".join(merges))

    vocab = json.load((tokenizer_path / "vocab.json").open())
    for token in tokens:
        logger.info(f"Added token {token} to the tokenizer.")
        vocab[token] = len(vocab)
    json.dump(vocab, (tokenizer_path / "vocab.json").open("w"), ensure_ascii=False)

    tokenizer_content = json.load((tokenizer_path / "tokenizer.json").open())
    tokenizer_content["model"]["vocab"] = vocab
    tokenizer_content["model"]["merges"] = merges[1:]
    json.dump(
        tokenizer_content,
        (tokenizer_path / "tokenizer.json").open("w"),
        indent=2,
        ensure_ascii=False,
    )


def copy_dir(src: Path, dst: Path):
    for file in src.iterdir():
        if file.is_dir():
            copy_dir(file, dst / file.name)
        else:
            file.rename(dst / file.name)


# Soft Max
def soft_max(arr):
    return arr.exp() / arr.exp().sum()

def get_first_word(original_text: str, predicted_text: str):
    predicted_text = predicted_text.replace(original_text, '')
    # Regex to find first word
    first_word = re.search(r'\b\w+\b', predicted_text)
    first_word = first_word.group() if first_word else None
    return first_word

def load_dataset_to_dataframe(*args, data_dir=None, dataset_types=['train', 'validation', 'test'], **kwargs):
    ds = load_dataset(*args, data_dir=data_dir, **kwargs)
    output = []
    for ds_type in dataset_types:
        output.append(ds[ds_type].to_pandas())
        output[-1]['Dataset Type'] = ds_type
    return pd.concat(output)




# Replace token from tokenizer
def replace_tokens(
    tokenizer: PreTrainedTokenizer,
    new_tokens: str | list[str],
    new_tokens_ids: int | list[int],
    temp_folder: str = '',
    delete_temp_folder: bool=True
):
    # Convert inputs to types for rest of script
    if isinstance(new_tokens, str): new_tokens = [new_tokens]
    if isinstance(new_tokens_ids, int): new_tokens_ids = [new_tokens_ids]

    # Assert we can match tokens with IDs
    assert len(new_tokens) == len(new_tokens_ids)

    # Save old tokenizer to temporary file
    if len(temp_folder) == 0:
        temp_folder = tempfile.gettempdir() + '/TOKENIZER_{}'.format(pd.Timestamp.now().strftime('%Y%m%d%H%M%S'))
    tokenizer.save_pretrained(temp_folder)

    # Delete all unnecessary files
    for file in os.listdir(temp_folder):
        if file.lower() not in ['tokenizer_config.json', 'tokenizer.json']:
            os.remove(f'{temp_folder}/{file}')
    
    # Updating tokenizer
    with open(f'{temp_folder}/tokenizer.json', 'r', encoding='utf-8') as f:
        t = json.load(f)

    merges = t['model']['merges'].copy()

    # Iterating for each new token added
    max_allowed_token_id = min(new_tokens_ids)
    while len(new_tokens) > 0:
        new_token = new_tokens[0]
        new_token_id = new_tokens_ids[0]

        # Obtain the old tokenization of the new_token to replace merges
        old_tokenization = tokenizer.tokenize(new_token)
        old_tokenization_ids = tokenizer.convert_tokens_to_ids(old_tokenization)
        old_token = tokenizer.convert_ids_to_tokens(new_token_id)


        # Iteratively remove tokens where their "TOKEN_ID" > min(NEW_TOKEN_IDS)
        #   If we're creating a new token ('martelo') with ID 3000 and the old tokenization of
        #   old_tokenization('martelo') = ['mart'(3102), 'el'(10)], we need to change the 
        #   token 'mart' to it's previous merge, because we'll try to access a token that has not yet
        #   been created when reaching the token with ID 3000
        max_old_tokenization_ids = max(old_tokenization_ids)
        while max_old_tokenization_ids > max_allowed_token_id:
            # Finding the index of the token to replace in "old_tokenization"
            #   (in our example, this would be the id of "mart" in ['mart', 'elo'], which is 0)
            replace_index = old_tokenization_ids.index(max_old_tokenization_ids)
            # Finding the merge that gives origin to that token found previously, so merge("mart") = ['m'(93), 'art'(434)] in our example
            old_token_merge = [(n, merge) for n, merge in enumerate(merges) if ''.join(merge) == old_tokenization[replace_index]][0]
            
            # Updating the list to include the "separated" tokens and continue with the cycle until no token exceeds the `min(NEW_TOKEN_IDS)`
            old_tokenization = old_tokenization[:replace_index] + old_token_merge[1] + old_tokenization[replace_index+1:]   # In the example, old_tokenization -> ['m', 'art', 'elo']
            old_tokenization_ids = tokenizer.convert_tokens_to_ids(old_tokenization)    # Following our example, old_tokenization_ids -> [93, 434, 10]
            max_old_tokenization_ids = max(old_tokenization_ids)

        # Update the new token (this takes care of old "spaces" becoming 'Ġ' and other nuances)
        new_token = ''.join(old_tokenization)

        # -----------------
        # Updating vocab
        # -----------------
        t['model']['vocab'].pop(old_token, None)  # Delete old token correspondent to new_token_id
        t['model']['vocab'][new_token] = new_token_id   # Add a new token with the new_token_id

        # -----------------
        # Updating Merges
        # -----------------
        # Find position of "old_token" in merges [merge that gives origin to old_token]
        old_token_merge_id = [n for n, merge in enumerate(t['model']['merges']) if ''.join(merge) == old_token][0]
        while len(old_tokenization) > 2:
            old_tokenization = [''.join(old_tokenization[:2])] + old_tokenization[2:]
            new_tokens_ids.append(min(new_tokens_ids) - 1)
            new_tokens.append(tokenizer.decoder.decode([old_tokenization[0]]))
            max_allowed_token_id = min(new_tokens_ids)
        t['model']['merges'][old_token_merge_id] = old_tokenization

        # Removing the first element to continue the cycle
        new_tokens = new_tokens[1:]
        new_tokens_ids = new_tokens_ids[1:]

    # -----------------
    # Writting tokenizer to file
    # -----------------

    with open(f'{temp_folder}/tokenizer.json', 'w', encoding='utf-8') as f:
        json.dump(t, f, indent=2, sort_keys=False, ensure_ascii=False)
    
    
    # -----------------
    # Initialize new Tokenizer
    # from File
    # -----------------
    new_tokenizer = AutoTokenizer.from_pretrained(temp_folder)

    # Delete temporary folder (if specified in argument)
    if delete_temp_folder:
        shutil.rmtree(temp_folder)
        temp_folder = None
    
    return {'tokenizer': new_tokenizer, 'tokenizer_path': temp_folder}


if __name__ == '__main__':
    import loader
    model, tokenizer = loader.load_model_and_tokenizer()
    new_tokenizer = replace_tokens(
        tokenizer,
        new_tokens=['funcionário', 'martelo'],
        new_tokens_ids=[49151, 31543],
    )
    new_tokenizer.encode('O meu martelo é bonito.')