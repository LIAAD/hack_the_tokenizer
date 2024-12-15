import pandas as pd
from typing import Callable, Union, Any
import tqdm
import transformers
from ..DatasetClass import ListDataset


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
        prediction_prompts: list[str],
        evaluation_method: Callable[['Benchmark', list[str]], Any],
        aggregation_method: Callable[[list[Any]], Any] = lambda x: sum(x) / len(x)
    ):
        self.name = name
        self.df = dataset
        self.prediction_prompts: ListDataset = ListDataset(prediction_prompts)
        self.eval_method = evaluation_method
        self.agg_method = aggregation_method
        self.evaluation_results = {}

    def run_eval(self, model_name: str, predictions: list[str]):
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
            'result': result,
            'results-raw': results
        }
        return result

    def __repr__(self):
        return f'<Benchmark[{self.name}]>'

class Benchmarks():
    config = {
        'parallel_tasks': True,
        'parallel_batch_size': None,
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
        'number_of_evaluations': 3,
        'max_new_tokens': 20,
    }

    def __init__(self, benchmarks: list[Benchmark]):
        self.benchmarks = benchmarks
        self.evaluation_results = {}

    def run(self, model: str, pipeline_kwargs={}):
        # Update each benchmark config before running benchmarks + Generate initial pipeline input prompt
        pipeline_inputs = []
        for bench in self.benchmarks:
            if bench.name not in self.config['dont_copy_config_benchmarks']:
                bench.config = self.config
            pipeline_inputs.extend(bench.prediction_prompts.to_list())
        
        generator = transformers.pipeline(
            model=model,
            torch_dtype=self.config.get('torch_dtype', None),
            model_kwargs=self.config.get('model_kwargs', None),
            **pipeline_kwargs
        )
        if generator.tokenizer.pad_token_id is None:
            generator.tokenizer.pad_token_id = generator.model.config.eos_token_id
        generation = []
        for gen in tqdm.tqdm(generator(
                ListDataset(pipeline_inputs),
                max_new_tokens=self.config.get('max_new_tokens', 5),
                batch_size=self.config.get('parallel_batch_size', None)
            ),
            desc=self.config.get('tqdm_desc', "<{MODEL}> Calculating inferences for inputs").format(MODEL=model),
            total=len(pipeline_inputs)
        ):
            generation.append(gen)

        outputs = {}
        for bench in self.benchmarks:
            outputs[bench.name] = [g['generated_text'] for g in generation[:len(bench.prediction_prompts)]]
            generation = generation[len(bench.prediction_prompts):]

        model_name = model.name_or_path
        if self.evaluation_results.get(model_name, None) is not None:
            return self.evaluation_results[model_name]
        results = {}
        for benchmark in self.benchmarks:
            eval_results = benchmark.run_eval(model, outputs[benchmark.name])
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