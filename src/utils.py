import re
import json
import torch
import logging
import pandas as pd

from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer


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
