import pandas as pd
from typing import Callable, Union, Any, Literal, overload
import tqdm
from hack_tokenizer.src.utils.DatasetClass import ListDataset, TextDataset
from torch.utils.data import DataLoader
from hack_tokenizer.src.utils.constants import TOKENIZER_TYPE, MODEL_TYPE


class Benchmark():
    config = {
        'parallel_batch_size': 8,
        'model_kwargs': {
            'do_sample': False,
            'temperature': None,
            'pad_token_id': 50257, # This is the tokenizer.eos_token_id
            # 'output_scores': True,            # INCLUDE LATER to get the probability distribution and use that as a benchmark
            # 'return_dict_in_generate': True,
            'return_legacy_cache': False
        },
        'gen_kwargs': {
            'num_beams': 1,
            'num_return_sequences': 1,
            'max_new_tokens': 5,
            'pad_token_id': None,
        },
        'number_of_evaluations': 1
    }
    
    def __init__(
        self,
        name: str,
        dataset: pd.DataFrame,
        prediction_prompts: list[str],
        evaluation_method: Callable[['Benchmark', dict], Any],
        aggregation_method: Callable[[list[Any]], Any] = lambda x: sum(x) / len(x)
    ):
        self.name = name
        self.df = dataset
        self.prediction_prompts: ListDataset = ListDataset(prediction_prompts)
        self.eval_method = evaluation_method
        self.agg_method = aggregation_method
        self.evaluation_results = {}

    @overload
    def get_benchmark_data(self, return_as: Literal['dataframe', 'df']) -> pd.DataFrame: ...
    @overload
    def get_benchmark_data(self, return_as: Literal['str']) -> str: ...
    @overload
    def get_benchmark_data(self, return_as: Literal['list']='list') -> list[str]: ...
    def get_benchmark_data(self, return_as: Literal['dataframe', 'df', 'list', 'str']='list') -> Union[pd.DataFrame, str, list[str]]:
        df = self.df
        df = df.select_dtypes(include='object').fillna('')
        if return_as in ('dataframe', 'df'):
            return df
        df['DATA'] = df.apply(lambda x: ' '.join(x).strip(), axis=1)
        df = df['DATA'].to_list()
        if return_as == 'list':
            return df
        return '\n'.join(df)

    def generate(self, model, tokenizer, encoding_tokenizer=None, generation_kwargs={}, store_generation_data=False):
        '''
            Method to generate predictions for given benchmark. This function can be overwriten by each BenchMark.
        '''
        # Initializing thee different tokenizers (we use the "original" tokenizer for encoding and the new tokenizer for decoding)
        decoder_tokenizer = tokenizer
        if encoding_tokenizer is None: encoding_tokenizer = tokenizer
        # Adding padding tokens to encoder tokenizer
        if encoding_tokenizer.pad_token is None:
            encoding_tokenizer.pad_token = encoding_tokenizer.eos_token
            encoding_tokenizer.pad_token_id = encoding_tokenizer.eos_token_id

        batch_size = self.config.get('parallel_batch_size', 1)
        dataloader = DataLoader(
            TextDataset(self.prediction_prompts.to_list(), encoding_tokenizer, batch_size),
            batch_size=batch_size,
            shuffle=False
        )
        gen_kwargs = self.config.get('gen_kwargs', {
            'num_beams': 1,
            'num_return_sequences': 1,
            'max_new_tokens': self.config.get('max_new_tokens', 5),
        })
        gen_kwargs.update(self.config.get('generation_kwargs', {})) # Second most priority arguments
        gen_kwargs.update(generation_kwargs)    # Most priority arguments
        # Force the generation to return dictionary output and include logits, scores and hidden_states
        gen_kwargs.update({
            'return_dict_in_generate': True,
            'output_logits': True,
            'output_scores': True,
            'output_hidden_states': True,
            'pad_token_id': encoding_tokenizer.pad_token_id if encoding_tokenizer.pad_token_id else encoding_tokenizer.eos_token_id,
        })
        generation = {
            'logits': [],
            'hidden_states': [],
            'scores': [],
            'generated_tokens': [],
            'generated_text': []
        }
        for batch in tqdm.tqdm(dataloader, desc=self.config.get('tqdm_desc', "<{MODEL}[{BENCHMARK}]> Calculating inferences for inputs").format(MODEL=model if isinstance(model, str) else model.name_or_path, BENCHMARK=self.name)):
            outputs = model.generate( # type: ignore
                input_ids=batch['input_ids'].squeeze(1).to(model.device), # type: ignore
                attention_mask=batch['attention_mask'].squeeze(1).to(model.device), # type: ignore
                **gen_kwargs
            )
            # Decode the input and generated sequences
            hidden_states = [[state.to('cpu') for state in output] for output in outputs.hidden_states]
            logits = [logit.to('cpu') for logit in outputs.logits]
            scores = [scores.to('cpu') for scores in outputs.scores]
            sequences = outputs.sequences.to('cpu')
            for i, generated_text in enumerate(decoder_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)):
                if store_generation_data:
                    generation['hidden_states'].append([[hidden_state[i] for hidden_state in token_hid_state] for token_hid_state in hidden_states])
                    generation['logits'].append([logit[i] for logit in logits])
                    generation['scores'].append([score[i] for score in scores])
                # generation[-1]['scores'] = [score[i] for score in scores]
                generation['generated_tokens'].append(sequences[i])
                generation['generated_text'].append(generated_text)
        # Remove all empty lists from output
        all_keys = list(generation.keys())
        for key in all_keys:
            if len(generation[key]) == 0:
                generation.pop(key)
        return generation

    def run_eval(self, model_name: str, predictions: dict[str, Any]):
        if self.evaluation_results.get(model_name, None) is not None:
            return self.evaluation_results[model_name]['result']
        self.evaluation_results[model_name] = {}
        results = []
        for _ in range(self.config['number_of_evaluations']):
            results.append(
                self.eval_method(self, predictions)
            )
        
        result = self.agg_method(results)
        self.evaluation_results[model_name] = {
            'model': model_name,
            'result': result,
            'results-raw': results
        }
        return result

    def __repr__(self):
        return f'<Benchmark[{self.name}]>'

class Benchmarks():
    config = {
        'parallel_batch_size': 1,
        'model_kwargs': {
            'do_sample': False,
            'temperature': None,
            'pad_token_id': 50257, # This is the tokenizer.eos_token_id
            # 'output_scores': True,            # INCLUDE LATER to get the probability distribution and use that as a benchmark
            # 'return_dict_in_generate': True,
            # 'return_legacy_cache': False
        },
        'dont_copy_config_benchmarks': [],
        'tqdm_description': '<{MODEL}> Running {benchmark_name} Benchmark',
        'number_of_evaluations': 1,
        'max_new_tokens': 20,
        'generation_kwargs': {}
    }

    def __init__(self, benchmarks: list[Benchmark]):
        self.benchmarks = benchmarks
        self.evaluation_results = {}

    def run(self, model, tokenizer, encode_tokenizer=None, generation_kwargs={}, store_generation_data=True):
        # Retrieve model name from `transformers model``
        model_name = model.name_or_path

        # Obtain the "decoder" and "encoder" tokenizers (in this generation we use a different encoding tokenizer then the decoder)
        decoder_tokenizer = tokenizer
        if encode_tokenizer is None: encode_tokenizer = tokenizer
        
        if self.evaluation_results.get(model_name, None) is not None:
            return self.evaluation_results[model_name]

        results = {}
        for benchmark in self.benchmarks:
            outputs = benchmark.generate(model, decoder_tokenizer, encode_tokenizer, generation_kwargs, store_generation_data)
            eval_results = benchmark.run_eval(model_name, outputs)
            results[benchmark.name] = {
                'result': eval_results,
                'results': benchmark.evaluation_results[model_name]['results-raw']
            }
        self.evaluation_results[model_name] = results
        return results
    
    def get_results(self):
        return self.evaluation_results

    @overload
    def get_benchmark_data(self, return_as: Literal['dataframe', 'df']) -> pd.DataFrame: ...
    @overload
    def get_benchmark_data(self, return_as: Literal['str']) -> str: ...
    @overload
    def get_benchmark_data(self, return_as: Literal['list']='list') -> list[str]: ...
    
    def get_benchmark_data(self, return_as: Literal['dataframe', 'df', 'list', 'str']='list') -> Union[pd.DataFrame, str, list[str]]:
        data = [b.get_benchmark_data(return_as=return_as) for b in self.benchmarks]
        if return_as in ('dataframe', 'df'):
            return pd.concat(data)  # type: ignore
        elif return_as == 'list':
            return [d for b in data for d in b] # type: ignore
        return '\n'.join(data)  # type: ignore

    def __repr__(self):
        out = '<<Benchmarks>>\n'
        for bench in self.benchmarks:
            out += f'    {bench.__repr__()}\n'
        return out[:-1]