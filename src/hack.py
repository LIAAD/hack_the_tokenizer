import os
import tqdm
import torch
import pickle

from typing import Literal, Union, Callable
from tokenizers import Tokenizer, trainers, models, pre_tokenizers, decoders

from .benchmark import BENCHMARKS
from . import utils

class TokenizerHack():

    def __init__(
        self,
        device='cuda',
        training_data=BENCHMARKS.get_training_data(),
        model=None,
        tokenizer=None
    ):
        self.device=device
        self.train_data = training_data
        self.model = model
        self.tokenizer = tokenizer


    def write_train_data(self, filename: str='trainer.txt'):
        with open(filename, 'w') as f:
            f.writelines(self.train_data)
        return filename


    def train_tokenizer(
        self,
        filename: str='trainer.txt',
        model=models.BPE(),
        pre_tokenizer=pre_tokenizers.ByteLevel(),
        decoder=decoders.ByteLevel(),
        trainer=trainers.BpeTrainer,
        trainer_args = [],
        trainer_kwargs= {
            'vocab_size': 10_000
        },
        keep_file: bool=False
    ):  
        self.write_train_data(filename)

        # Initialize the tokenizer
        tokenizer = Tokenizer(model)
        tokenizer.pre_tokenizer = pre_tokenizer
        tokenizer.decoder = decoder

        tokenizer_trainer = trainer(*trainer_args, **trainer_kwargs)
        tokenizer.train(files=[filename], trainer=tokenizer_trainer)

        if not keep_file: os.remove(filename)
        self.trained_tokenizer = tokenizer
        return tokenizer


    def update_tokens_embed(
        self,
        model,
        original_tokenizer,
        new_tokens,
        new_tokens_ids,
        method: Union[Literal['min', 'mean', 'mean', 'avg'], Callable[[list[list[int]]], list[int]]]='mean'
    ):
        method_func = {
            'mean':     lambda x: torch.mean(x, dim=0),
            'avg':      lambda x: torch.mean(x, dim=0),
            'min':      lambda x: torch.min(x, dim=0),
            'max':      lambda x: torch.max(x, dim=0),
            'quantile': lambda x, q: torch.quantile(x, q, dim=0)  # TODO: Find a way to integrate quantile here
        }
        if isinstance(method, str):
            method = method_func[method]

        # Get the embedding layer
        embedding_layer = model.get_input_embeddings()
        embedding_layer.weight.requires_grad = False  # turn off gradients

        for new_token, new_token_id in zip(new_tokens, new_tokens_ids):
            old_token_ids = original_tokenizer.encode(new_token, return_tensors='pt').to(self.device)
            # Compute mean embedding of the new tokenizer
            new_embed = method_func(embedding_layer(old_token_ids))
            embedding_layer.weight[new_token_id] = new_embed
        
        embedding_layer.weight.requires_grad = True  # turn gradients back on
        return model


    def hack(self, model, tokenizer, embed_updt_method='mean'):
        '''
            NOTE: This changes the model in-place
        '''
        # # Find the new tokenizers given the initial training data
        # new_tokenizer = self.train_tokenizer(
        #     trainer_kwargs={'vocab_size': len(tokenizer.vocab)}
        # )

        # # Find which vocab is in "new_tokenizer" and not in "tokenizer"
        # new_vocab = new_tokenizer.get_vocab()
        old_vocab = tokenizer.vocab
        # # Delete all tokens in "old_vocab" from "new_vocab"
        # for token in old_vocab.keys():
        #     new_vocab.pop(token, None)
        
        # # Update the new vocab to "normalize" the tokens (e.g: "Ġexercer" will become " exercer". 
        # #   This ensures that in the following steps we treat with the text correctly)
        # new_vocab = {
        #     token: token_id 
        #     for token, token_id in [(new_tokenizer.decode([t_id]), t_id) for t_id in new_vocab.values()]
        # }

        with open('NEW_VOCAB.PICKLE', 'rb') as f:
            new_vocab = pickle.load(f)
        new_tokens = list(new_vocab.keys())[:1000]  # Adding 1000 new tokens
        new_tokens_ids = list(range(len(old_vocab)-1, len(old_vocab)-len(new_tokens)-1, -1))

        # Add the new tokens to the tokenizer
        hacked_tokenizer = utils.replace_tokens(
            tokenizer,
            new_tokens,
            new_tokens_ids,
            delete_temp_folder=False
        )
        hacked_tokenizer, hacked_tokenizer_folder = hacked_tokenizer['tokenizer'], hacked_tokenizer['tokenizer_path']


        # Calculate the new embedding weights given the new tokenizer
        new_model = self.update_tokens_embed(
            model=model,
            original_tokenizer=tokenizer,
            new_tokens=new_tokens,
            new_tokens_ids=new_tokens_ids,
            method=embed_updt_method
        )
        return {
            "model": new_model,
            'tokenizer': hacked_tokenizer,
            'tokenizer_path': hacked_tokenizer_folder
        }
