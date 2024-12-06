import pandas as pd
from typing import Callable, Union, Any, Optional, Literal
import torch
import transformers


TOKENIZER_TYPE = Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]
MODEL_TYPE = transformers.AutoModel

class Benchmark():
    config = {
        'parallel_tasks': True,
        'parallel_batch_size': 22,
        'model_kwargs': {
            'do_sample': False,
            'temperature': None,
            'pad_token_id': 50257, # This is the tokenizer.eos_token_id
            # 'output_scores': True,            # INCLUDE LATER to get the probability distribution and use that as a benchmark
            # 'return_dict_in_generate': True,
            'return_legacy_cache': False
        },
        'number_of_evaluations': 3
    }
    
    def __init__(
        self,
        name: str,
        dataset: pd.DataFrame,
        evaluation_method: Callable[['Benchmark', MODEL_TYPE, TOKENIZER_TYPE], Any],
        aggregation_method: Callable[[list[Any]], Any] = lambda x: sum(x) / len(x)
    ):
        self.name = name
        self.df = dataset
        self.eval_method = evaluation_method
        self.agg_method = aggregation_method
        self.evaluation_results = {}
    
    def run_eval(self, model: MODEL_TYPE, tokenizer: TOKENIZER_TYPE):
        model_name = model.name_or_path
        if self.evaluation_results.get(model_name, None) is not None:
            return self.evaluation_results[model_name]['result']
        self.evaluation_results[model_name] = {}
        results = []
        for _ in range(self.config['number_of_evaluations']):
            results.append(
                self.eval_method(self, model, tokenizer)
            )
        
        result = self.agg_method(results)
        self.evaluation_results[model_name] = {
            'result': result,
            'results-raw': results
        }
        return result

    def __repr__(self):
        return f'<Benchmark[{self.name}]>'

class Benchmarks():
    config = {
        'parallel_tasks': True,
        'parallel_batch_size': 22,
        'model_kwargs': {
            'do_sample': False,
            'temperature': None,
            'pad_token_id': 50257, # This is the tokenizer.eos_token_id
            # 'output_scores': True,            # INCLUDE LATER to get the probability distribution and use that as a benchmark
            # 'return_dict_in_generate': True,
            'return_legacy_cache': False
        },
        'no_config_copy_benchmarks': [],
        'tqdm_description': '<{MODEL}> Running {benchmark_name} Benchmark',
        'number_of_evaluations': 3
    }

    def __init__(self, benchmarks: list[Benchmark]):
        self.benchmarks = benchmarks
        self.evaluation_results = {}

    def run(self, model: MODEL_TYPE, tokenizer: TOKENIZER_TYPE):
        # Update each benchmark config before running benchmarks
        for bench in self.benchmarks:
            if bench.name not in self.config['no_config_copy_benchmarks']: bench.config = self.config
    
        model_name = model.name_or_path
        if self.evaluation_results.get(model_name, None) is not None:
            return self.evaluation_results[model_name]
        results = {}
        for benchmark in self.benchmarks:
            eval_results = benchmark.run_eval(model, tokenizer)
            results[benchmark.name] = {
                'result': eval_results,
                'results': benchmark.evaluation_results
            }
        self.evaluation_results[model_name] = results
        return results
    
    def get_results(self):
        return self.evaluation_results

    def __repr__(self):
        out = '<<Benchmarks>>\n'
        for bench in self.benchmarks:
            out += f'    {bench.__repr__()}\n'
        return out[:-1]