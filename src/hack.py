import os
import tqdm
import torch
import pickle

from src import utils, loader, hack
from src.DatasetClass import ListDataset, TextDataset
from torch.utils.data import DataLoader

from typing import Literal, Union, Callable
from tokenizers import Tokenizer, trainers, models, pre_tokenizers, decoders

from src.benchmark import BENCHMARKS
from src import utils

class TokenizerHack():

    def __init__(
        self,
        device='cuda',
        training_data: list[str]=BENCHMARKS.get_training_data(),
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


class ModelHacker():

    def __init__(self, dataset: str | list[str], batch_size: int, learning_rate: float=1e-6):
        self.dataset = [dataset] if isinstance(dataset, str) else dataset
        self.batch_size = batch_size
        self.lr = learning_rate

    def _get_new_tokens(self, vocab_size: int, ignore_tokens: list[str]=[], remove_contained_tokens: bool=True):
        '''
            Given a `vocab_size` and `ignore_tokens`, will return a list[str] where each element represents a token.
            The length of this output matches `vocab_size`.

            **Note**: The output won't contain any token given as input in `ignore_tokens`. The usually utilization of this function should be
            to pass `tokenizer.vocab.keys()` to the "ignore_tokens" argument.

            Parameters
            ----------

            vocab_size: int
                Expected length for the output list containing the new tokens.
            
            ignore_tokens: list[str], default=[]
                List containing tokens to ignore in the output tokens

            remove_contained_tokens: bool, default=True
                Flag to control wether or not to remove tokens contained in the `ignore_tokens` list.
        '''
        new_tokens: set[str] = set([])
        extra = 0
        while len(new_tokens) < vocab_size:
            tokenizer__ = TokenizerHack(training_data=self.dataset).train_tokenizer(trainer_kwargs={'vocab_size': 2*vocab_size + extra})

            # Step 2. Find tokens in `pt_tokenizer` not in 
            new_tokens: set[str] = set([tokenizer__.decode([x]) for x in range(vocab_size)])
            new_tokens = new_tokens.difference(set(ignore_tokens))
            extra += 100
    

        # Remove the tokens which may be "contained" in any of the original tokens (for instance, "publ" is contained in "publico" so "publ" will be removed)
        if remove_contained_tokens:
            __new_tokens = []
            for new_token in tqdm.tqdm(new_tokens, total=len(new_tokens)):
                add_new_token = True
                for token in ignore_tokens:
                    if token.startswith(new_token):
                        add_new_token = False
                        break
                if add_new_token: __new_tokens.append(new_token)
            new_tokens = list(set(__new_tokens))
        return new_tokens

    def _initialize_embeddings(self, model, tokenizer, new_tokens: list[str]):
        # ----------------------------------------------------
        #               Update Model Vocabulary               
        # ----------------------------------------------------
        # Save the original tokenizations
        original_tokenization = {t: tokenizer.encode(t) for t in new_tokens}    # Necessary for the training bellow
        tokenizer.add_tokens(list(new_tokens))
        model.resize_token_embeddings(len(tokenizer))

        # Step 4. Calculate the new embeddings for the new tokens
        embed = model.get_input_embeddings().weight.clone().to('cpu')
        new_embed = model.get_input_embeddings()

        # Initialize the embedding using the weighted average model
        K = 1.5 # Tested for values [1, 2, 3, 4, 5, 0.9, 0.8, 1.1, 1.2, ..., 1.6] and the best was 1.5 with (3.13%)
        with torch.no_grad():
            for new_token in tqdm.tqdm(new_tokens, desc='Initializing the embeddings for the new_tokens'):
                new_token_id = tokenizer.encode(new_token)[0]
                # Find the old embedding for the token
                tokenization = original_tokenization[new_token]
                token_embed = torch.stack([embed[t_id] for t_id in tokenization]).to(model.device.type)
                # Calculating the embedding weights
                embedding_weights = torch.asarray([K**i if K**i < 2**64 else 0 for i in range(token_embed.shape[0], 0, -1)]).to(model.device.type)
                # embedding_weights = torch.asarray([K**i for i in range(token_embed.shape[0], 0, -1)]).to(DEVICE)
                embedding_weights = embedding_weights / embedding_weights.sum()

                # Create a new token embed using the weighted average of the embeddings
                new_token_embed = torch.sum(token_embed * embedding_weights[:, None], dim=0)
                # new_token_embed = token_embed[0]
                # Update embedding of the new_token in the hacked_model
                _ = new_embed.weight[new_token_id].data.copy_(new_token_embed)
        return model, tokenizer

    def train(self, model, tokenizer, num_tokens: int, new_tokens: list[str]=None):
        '''
            Parameters
            ----------
                model:
                    Model to which apply the training to

                tokenizer:
                    Tokenizer for the model given
                
                num_tokens: int
                    Number of tokens to add to tokenizer and model. This should be a value greater than 0
        '''
        if new_tokens is None:
            new_tokens = self._get_new_tokens(num_tokens, ignore_tokens=[tokenizer.decode([x]) for x in range(len(tokenizer))])

        # Initialize embeddings in model and tokenizer
        model, tokenizer = self._initialize_embeddings(model, tokenizer, new_tokens)

        # ----------------------------------------------------
        #     Update weights of Embeddings of new_tokens      
        # ----------------------------------------------------
        # Step 4.2 Using the training phrases to update the embedding weights
        new_embed = model.get_input_embeddings()
        for new_token in tqdm.tqdm(new_tokens, desc='Updating the embeddings for the new tokens'):
            new_token_id = tokenizer.convert_tokens_to_ids(new_token)
            new_token = tokenizer.decode(new_token_id)
            phrases_to_generate_new_token = [p for phrase in self.dataset for p in phrase.split(new_token)[:-1] if new_token in phrase and len(p) > 0]

            if len(phrases_to_generate_new_token) == 0: continue

            # Creating the Batched dataset (to run generation for multiple phrases at the same time)
            dataloader = DataLoader(
                TextDataset(phrases_to_generate_new_token, tokenizer, max_length=max(len(tokenizer.tokenize(x)) for x in phrases_to_generate_new_token)),
                batch_size=self.batch_size,
                shuffle=False
            )
            # Process the batches
            for batch in tqdm.tqdm(dataloader,  desc=f'  Generating tokens for new_token=`{new_token}` ', leave=False):
                # Move batch tensors to the correct device
                input_ids = batch['input_ids'].squeeze(1).to(model.device.type)
                attention_mask = batch['attention_mask'].squeeze(1).to(model.device.type)

                # Generate text
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1,
                    num_beams=1,
                    num_return_sequences=1,
                    return_dict_in_generate=True,
                    output_logits=True,
                    output_scores=True,
                    output_hidden_states=True,
                    pad_token_id=tokenizer.pad_token_id,
                    **dict(top_p=None, top_k=None, temperature=None, do_sample=False)
                )

                # Extract the generated sequences and their scores
                predicted_logits = outputs.logits

                # Decode the input and generated sequences
                input_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                with torch.no_grad():
                    for i in range(len(input_texts)):
                        logits = predicted_logits[0][i]
                        logit_gradient = logits.max() - logits[new_token_id]
                        embed_out = outputs.hidden_states[0][-1][i][-1]
                        # normalize embed_out
                        embed_out = embed_out / embed_out.norm()

                        embed_in = new_embed.weight[new_token_id]

                        # Update the embedding table
                        _ = new_embed.weight[new_token_id].data.copy_((embed_in + logit_gradient * embed_out * self.lr).to(model.device.type))



if __name__ == '__main__':

    DEVICE                  = 'cuda'
    GENERATION_BATCH_SIZE   = 8
    MODEL                   = 'Qwen/Qwen2.5-1.5B-Instruct'
    MODEL_GEN_KWARGS = dict(top_p=None, top_k=None, temperature=None, do_sample=False)

    m_hacker = ModelHacker(
        batch_size=GENERATION_BATCH_SIZE,
        dataset=BENCHMARKS.get_training_data(),
        learning_rate=1e-6
    )

    model, tokenizer = loader.load_model_and_tokenizer(
        model_name=MODEL,
        device=DEVICE,
        model_kwargs = { 'torch_dtype': torch.bfloat16},
        tokenizer_kwargs={'padding_side': 'left'}
    )
    m_hacker.train(
        model,
        tokenizer,
        num_tokens=5000
    )