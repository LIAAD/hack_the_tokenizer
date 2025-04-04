import pandas as pd
from typing import Callable, Union, Any, Literal
import tqdm
import transformers
from ..DatasetClass import ListDataset, TextDataset
from torch.utils.data import DataLoader


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
        'number_of_evaluations': 1
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

    def run_eval(self, model_name: str, predictions: list[str], tokenizer):
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

    def run(self, model, tokenizer, generation_kwargs={}):
        # Update each benchmark config before running benchmarks + Generate initial pipeline input prompt
        pipeline_inputs = []
        for bench in self.benchmarks:
            if bench.name not in self.config['dont_copy_config_benchmarks']:
                bench.config = self.config
            pipeline_inputs.extend(bench.prediction_prompts.to_list())
        
        batch_size = self.config.get('parallel_batch_size', 1)
        dataloader = DataLoader(
            TextDataset(pipeline_inputs, tokenizer, batch_size),
            batch_size=batch_size,
            shuffle=False
        )
        generation = []
        gen_kwargs = {
            'num_beams': 1,
            'num_return_sequences': 1,
            'max_new_tokens': self.config.get('max_new_tokens', 5),
            'pad_token_id': tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
        }
        gen_kwargs.update(self.config.get('generation_kwargs', {})) # Second most priority arguments
        gen_kwargs.update(generation_kwargs)    # Most priority arguments
        
        for batch in tqdm.tqdm(dataloader, desc=self.config.get('tqdm_desc', "<{MODEL}> Calculating inferences for inputs").format(MODEL=model if isinstance(model, str) else model.name_or_path)):
            outputs = model.generate(
                input_ids=batch['input_ids'].squeeze(1).to(model.device),
                attention_mask=batch['attention_mask'].squeeze(1).to(model.device),
                **gen_kwargs
            )
            # Decode the input and generated sequences
            generation.extend([{'generated_text': x} for x in tokenizer.batch_decode(outputs, skip_special_tokens=True)])

        outputs = {}
        for bench in self.benchmarks:
            outputs[bench.name] = [g['generated_text'] for g in generation[:len(bench.prediction_prompts)]]
            generation = generation[len(bench.prediction_prompts):]

        model_name = model.name_or_path
        if self.evaluation_results.get(model_name, None) is not None:
            return self.evaluation_results[model_name]
        results = {}
        for benchmark in self.benchmarks:
            eval_results = benchmark.run_eval(model_name, outputs[benchmark.name], tokenizer=tokenizer)
            results[benchmark.name] = {
                'result': eval_results,
                'results': benchmark.evaluation_results[model_name]['results-raw']
            }
        self.evaluation_results[model_name] = results
        return results
    
    def get_results(self):
        return self.evaluation_results

    def get_training_data(self, return_as: Literal['dataframe', 'df', 'list', 'str']='list'):
        training_data = pd.concat([b.df for b in self.benchmarks]).reset_index(drop=True)
        training_data = training_data.select_dtypes(include='object').fillna('').drop(columns=['Dataset Type', 'prediction_prompts'])
        if return_as in ('dataframe', 'df'):
            return training_data
        training_data['DATA'] = training_data.apply(lambda x: ' '.join(x).strip(), axis=1)
        training_data = training_data['DATA'].to_list()
        if return_as == 'list':
            return training_data
        return '\n'.join(training_data)

    def __repr__(self):
        out = '<<Benchmarks>>\n'
        for bench in self.benchmarks:
            out += f'    {bench.__repr__()}\n'
        return out[:-1]