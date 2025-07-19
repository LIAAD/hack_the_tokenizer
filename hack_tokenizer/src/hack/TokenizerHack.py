import os

from tokenizers import Tokenizer, trainers, models, pre_tokenizers, decoders
from hack_tokenizer.src.benchmark import BENCHMARKS


class TokenizerHack():

    def __init__(
        self,
        device='cuda',
        training_data: list[str]=BENCHMARKS.get_benchmark_data(),
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
        tokenizer.pre_tokenizer = pre_tokenizer # type: ignore
        tokenizer.decoder = decoder # type: ignore

        tokenizer_trainer = trainer(*trainer_args, **trainer_kwargs)
        tokenizer.train(files=[filename], trainer=tokenizer_trainer)

        if not keep_file: os.remove(filename)
        self.trained_tokenizer = tokenizer
        return tokenizer

