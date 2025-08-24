import re
import tqdm
import torch
import logging
import numpy as np
import transformers
import sys
sys.path.insert(0, '/home/yali/MEGA/Hack The Tockenizer')

from ..utils import loader
from ..utils.DatasetClass import TextDataset
from torch.utils.data import DataLoader

from typing import Literal, Union, Callable, Optional

from .TokenizerHack import TokenizerHack

logger = logging.getLogger(__name__)

class ModelHacker():

    def __init__(self, dataset: str | list[str], batch_size: int, learning_rate: float=1e-6):
        self.dataset = [dataset] if isinstance(dataset, str) else dataset
        self.batch_size = batch_size
        self.lr = learning_rate
        self.new_tokens: list[str] = []

    def _get_new_tokens(self, vocab_size: int, ignore_tokens: list[str]=[], remove_contained_tokens: bool=True, show_progress: bool=True) -> list[str]:
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
        new_tokens_set: set[str] = set([])
        extra = int(np.sqrt(vocab_size))
        new_tokens_size = 0
        trial_count = -1
        while new_tokens_size < vocab_size:
            trial_count += 1
            if show_progress:
                logger.info(f'<trial no.{trial_count}> Calculating tokenizers')
            tokenizer__ = TokenizerHack(training_data=self.dataset).train_tokenizer(trainer_kwargs={'vocab_size': 2*vocab_size + extra, 'show_progress': show_progress})

            # Step 2. Find tokens in `pt_tokenizer` not in 
            new_tokens_set: set[str] = set([tokenizer__.decode([x]) for x in range(tokenizer__.get_vocab_size())])
            new_tokens_set = new_tokens_set.difference(set(ignore_tokens))
            extra += (vocab_size - len(new_tokens_set)) * 2 # Times two to speed-up reaching the target number of tokens
            if new_tokens_size >= len(new_tokens_set):
                break
            new_tokens_size = len(new_tokens_set)
    
        new_tokens: list[str] = list(new_tokens_set)

        # Remove the tokens which may be "contained" in any of the original tokens (for instance, "publ" is contained in "publico" so "publ" will be removed)
        if remove_contained_tokens:
            __new_tokens = []
            for new_token in tqdm.tqdm(new_tokens_set, total=len(new_tokens_set), desc='Removing tokens "contained" within any token of original tokenizer'):
                add_new_token = True
                for token in ignore_tokens:
                    if token.startswith(new_token):
                        add_new_token = False
                        break
                if add_new_token: __new_tokens.append(new_token)
            new_tokens = list(set(__new_tokens))
        return new_tokens[:vocab_size]

    def _get_reduction_method(
        self,
        method: Union[Literal['min', 'max', 'mean', 'avg', 'quantile'], str] = 'mean'
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Returns the appropriate reduction function based on method string.
        Handles both basic methods and parametric methods like 'quantile(0.5)'/'weighted_drop(1.5)'.
        """
        # Validate method format
        if isinstance(method, str) and '(' in method:
            if not re.fullmatch(r'(quantile|weighted_drop)\(\d*\.?\d+\)', method):
                raise ValueError(f"Invalid method format: {method}. Expected 'quantile(x)' or 'weighted_drop(x)'")
            
            method_type, param_str = method.split('(')
            param = float(param_str[:-1])  # Remove trailing ')'
            
            if method_type == 'weighted_drop':
                def reducer(token_embed: torch.Tensor) -> torch.Tensor:
                    n = token_embed.shape[0]
                    weights = (param ** torch.arange(n, 0, -1, device=token_embed.device))
                    weights = weights / weights.sum()
                    return torch.sum(token_embed * weights[:, None], dim=0)
                return reducer
                
            elif method_type == 'quantile':
                return lambda x: torch.quantile(x, q=param, dim=0)
        
        # Basic methods
        reducers = {
            'mean': lambda x: torch.mean(x, dim=0),
            'avg': lambda x: torch.mean(x, dim=0),
            'min': lambda x: torch.min(x, dim=0).values,
            'max': lambda x: torch.max(x, dim=0).values,
            'quantile': lambda x: torch.quantile(x, q=0.5, dim=0),  # Default median
        }
        
        if method not in reducers:
            raise ValueError(f"Unknown method: {method}")
            
        return reducers[method]
    def _initialize_embeddings(
        self,
        model, tokenizer,
        new_tokens: list[str],
        method: Union[Literal['min', 'mean', 'mean', 'avg', 'quantile({number})', 'weighted_drop({number})'], str]='mean'
    ):
        """
        Initializes embeddings according to the specified method
        
        Parameters
        ----------

        method: Literal['min', 'mean', 'mean', 'avg', 'quantile({number})', 'weighted_drop({number})], default='mean'
            Either a basic statistic ('min', 'max', 'mean', 'avg') or:
            - 'weighted_drop(K)' where K is the decay factor.
            - 'quantile(P)' where p is the "quantile" factor.
        """
        # Before starting, validate the method provided
        method_func = self._get_reduction_method(method)

        # ----------------------------------------------------
        #               Update Model Vocabulary               
        # ----------------------------------------------------
        # Save the original tokenizations
        original_tokenization = {t: tokenizer.encode(t) for t in new_tokens}    # Necessary for the training bellow
        tokenizer.add_tokens(list(new_tokens))
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

        # Step 4. Calculate the new embeddings for the new tokens
        embed: torch.nn.modules.sparse.Embedding = model.get_input_embeddings().weight.clone().to('cpu')
        new_embed: torch.nn.modules.sparse.Embedding = model.get_input_embeddings()

        # Initialize the embedding using the weighted average model (Meaning Tokenization[new_t] = [t1, t2, ..., tn] => Embed[new_t] =  where )
        with torch.no_grad():
            for new_token in tqdm.tqdm(new_tokens, desc='Initializing the embeddings for the new_tokens'):
                new_token_id = tokenizer.encode(new_token)[0]

                # Find Embedding for all tokens of old tokenization
                tokenization = original_tokenization[new_token]
                token_embed = torch.stack([embed[t_id] for t_id in tokenization]).to(model.device.type) # type: ignore
    
                new_token_embed = method_func(token_embed)
                # Update embedding of the new_token in the hacked_model
                _ = new_embed.weight[new_token_id].data.copy_(new_token_embed)
        return model, tokenizer

    def hack(
        self,
        model, tokenizer,
        encoding_tokenizer,
        num_tokens: int,
        new_tokens: Optional[list[str]]=None,
        embed_initializer_method: Union[Literal['min', 'mean', 'mean', 'avg', 'quantile({number})', 'weighted_drop({number})'], str]='weighted_drop(1.5)',
        show_progress: bool=False,
        train: bool=True,
        train_kwargs: dict={
            'dataset': None,
            'batch_size': None,
        }
    ):
        '''
        Parameters
        ----------
        model:
            Model to which apply the training to

        tokenizer:
            Tokenizer for the model given
        
        num_tokens: int
            Number of tokens to add to tokenizer and model. This should be a value greater than 0
        
        new_tokens: list[str], default=None
            This argument...

        show_progress: bool, default=False
            This argument...

        train: bool, default=True
            Flag which specifies if the embeddings should go through training or not.

        train_kwargs: dic, default={'dataset': None, 'batch_size': None}
            This argument...
        '''
        if new_tokens is None:
            self.new_tokens = self._get_new_tokens(num_tokens, ignore_tokens=[tokenizer.decode([x]) for x in range(len(tokenizer))], show_progress=show_progress)
        else: self.new_tokens = new_tokens

        # Initialize embeddings in model and tokenizer
        model, tokenizer = self._initialize_embeddings(model, tokenizer, self.new_tokens, embed_initializer_method)

        # Adding padding tokens to encoder tokenizer
        if encoding_tokenizer.pad_token is None:
            encoding_tokenizer.pad_token = encoding_tokenizer.eos_token
            encoding_tokenizer.pad_token_id = encoding_tokenizer.eos_token_id
        
        # Adding padding tokens to tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if train:
            train_kwargs.setdefault('show_progress', show_progress)
            self.train(model=model, tokenizer=tokenizer, encoding_tokenizer=encoding_tokenizer, **train_kwargs)
        return model, tokenizer

    def train(
        self,
        model, tokenizer, 
        encoding_tokenizer,
        dataset: Optional[Union[str, list[str]]]=None,
        tokens: Optional[list[str]]=None, 
        batch_size: Optional[int]=None,
        show_progress: bool=True,
    ):
        # If no token was specified to train on, all added tokens will be considered
        if tokens is None:
            tokens = self.new_tokens
        # Setting up configuration
        dataset = self.dataset if dataset is None else [dataset] if isinstance(dataset, str) else dataset
        batch_size = self.batch_size if batch_size is None else batch_size

        # ----------------------------------------------------
        #     Update weights of Embeddings of new_tokens      
        # ----------------------------------------------------
        # Step 4.2 Using the training phrases to update the embedding weights
        new_embed: torch.nn.modules.sparse.Embedding = model.get_input_embeddings()
        iterator = tqdm.tqdm(tokens, desc='Updating the embeddings for the new tokens') if show_progress else tokens
        for new_token in iterator:
            new_token_id: int = tokenizer.convert_tokens_to_ids(new_token)
            new_token: str = tokenizer.decode(new_token_id)
            phrases_to_generate_new_token = [p for phrase in dataset for p in phrase.split(new_token)[:-1] if new_token in phrase and len(p) > 0]

            if len(phrases_to_generate_new_token) == 0: continue

            # Creating the Batched dataset (to run generation for multiple phrases at the same time)
            dataloader = DataLoader(
                TextDataset(phrases_to_generate_new_token, encoding_tokenizer, max_length=max(len(tokenizer.tokenize(x)) for x in phrases_to_generate_new_token)),
                batch_size=self.batch_size,
                shuffle=False
            )
            # Process the batches
            batch_iterator = tqdm.tqdm(dataloader,  desc=f'  Generating tokens for new_token=`{new_token}` ', leave=False) if show_progress else dataloader
            for batch in batch_iterator:
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
        return model

    @classmethod
    def prompt(cls, model, tokenizer, encoding_tokenizer, content: list[str] | str, max_new_tokens: int, stop_words: list[str], temperature: Optional[float]=None, print_response: bool=True):
        model_gen_kwargs = dict(top_p=None, top_k=None, temperature=None, do_sample=False) if temperature is None else dict(do_sample=True, temperature=temperature)
        model_gen_kwargs.update({'max_new_tokens': 1, 'pad_token_id': tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id})

        # Move batch tensors to the correct device
        inputs = encoding_tokenizer(content, return_tensors='pt')
        input_ids = inputs['input_ids'].to(model.device.type)
        attention_mask = inputs['attention_mask'].to(model.device.type)
    
        # Generate text
        outputs = []
        while len(outputs) < max_new_tokens:
            if len(outputs) > 1:
                inputs = encoding_tokenizer(content + ''.join(outputs), return_tensors='pt')
                input_ids = inputs['input_ids'].to(model.device.type)
                attention_mask = inputs['attention_mask'].to(model.device.type)
            new_token = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **model_gen_kwargs
            )[0][-1].item()
            outputs.append(tokenizer.decode(new_token))
            if print_response: print(outputs[-1], end='')
            # Check if any of the stop words have been generated
            for stop_word in stop_words:
                if ''.join(outputs[-1]).endswith(stop_word):
                    break
        
        return outputs


if __name__ == '__main__':
    from ..benchmark import BENCHMARKS, CalamePT, SupergluePTPT

    DEVICE                  = 'cpu'
    GENERATION_BATCH_SIZE   = 8
    # MODEL                   = 'Qwen/Qwen2.5-1.5B-Instruct'
    MODEL                   = 'HuggingFaceTB/SmolLM-135M'
    MODEL_GEN_KWARGS = dict(top_p=None, top_k=None, temperature=None, do_sample=False)

    m_hacker = ModelHacker(
        batch_size=GENERATION_BATCH_SIZE,
        dataset=BENCHMARKS.get_benchmark_data(),
        learning_rate=1e-6
    )

    model, tokenizer = loader.load_model_and_tokenizer(
        model_name=MODEL,
        device=DEVICE,
        model_kwargs = { 'torch_dtype': torch.bfloat16},
        tokenizer_kwargs={'padding_side': 'left'}
    )
    encoding_tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, padding_side='left')
    m_hacker.hack(
        model, tokenizer,
        tokenizer,
        num_tokens=1000,
        embed_initializer_method='weighted_drop(1.5)',
        show_progress=True,
        train=True,
        train_kwargs={
            'dataset': CalamePT.CalamePT().prediction_prompts.to_list(), # Only use CalamePT for dataset
        }
    )