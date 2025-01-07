import tqdm
import torch

from typing import Literal, Union, Callable
from tokenizers import Tokenizer, trainers, models, pre_tokenizers

from .benchmark import BENCHMARKS


class TokenizerHack():

    def __init__(
        self,
        device='cuda',
        training_data=BENCHMARKS.get_training_data()
    ):
        self.device=device
        self.train_data = training_data

    def write_train_data(self, filename: str='trainer.txt'):
        with open(filename, 'w') as f:
            f.writelines(self.train_data)
        return filename

    def get_tokenizer(
        self,
        filename: str='trainer.txt',
        model=models.BPE(),
        pre_tokenizer=pre_tokenizers.ByteLevel(),
        trainer=trainers.BpeTrainer,
        trainer_args = [],
        trainer_kwargs= {
            'vocab_size': 10_000
        }
    ):  
        self.write_train_data(filename)

        # Initialize the tokenizer
        tokenizer = Tokenizer(model)
        tokenizer.pre_tokenizer = pre_tokenizer

        tokenizer_trainer = trainer(*trainer_args, **trainer_kwargs)
        tokenizer.train(files=[filename], trainer=tokenizer_trainer)
        return tokenizer

    def get_new_embedding(
        self,
        embed_matrix: torch.nn.modules.sparse.Embedding,
        old_tokenizer: Tokenizer,
        new_tokenizer: Tokenizer,
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

        new_weights = []
        for token_id in tqdm.trange(new_tokenizer.get_vocab_size(), desc='Calculating new `Embedding Table` Weights'):
            token_name = new_tokenizer.decode([token_id])
            new_weights.append(method(embed_matrix._parameters['weight'][old_tokenizer.encode(token_name)]))

        # After calculating the new weights, create the new embedding layer with those weights
        new_embed = torch.nn.modules.Embedding(new_tokenizer.get_vocab_size(), embed_matrix.embedding_dim)
        new_embed._parameters['weight'] = torch.nn.parameter.Parameter(torch.stack(new_weights))
        return new_embed

    def hack(self, model, tokenizer, embed_updt_method='mean'):
        '''
            NOTE: This changes the model in-place
        '''
        # Find the new tokenizers given the initial training data
        new_tokenizer = self.get_tokenizer(
            trainer_kwargs={'vocab_size': len(tokenizer.vocab)}
        )

        # Calculate the new embedding weights given the new tokenizer
        new_embed = self.get_new_embedding(
            model.get_input_embeddings(),
            old_tokenizer=tokenizer,
            new_tokenizer=new_tokenizer,
            method=embed_updt_method
        )

        # Update the model's embedding table with the new one
        model.set_input_embeddings(new_embed)
        # model.set_output_embeddings(new_embed)  # TODO: Find how to correctly swap the output embeddings, this doesn't seem to produce the desired outcome
        return {
            "model": model,
            "tokenizer": new_tokenizer
        }
