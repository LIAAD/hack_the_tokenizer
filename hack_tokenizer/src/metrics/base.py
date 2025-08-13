from typing import Any, Optional, Union
import os

class Metric:
    def __init__(self, data: Union[str, list[str]], name: Optional[str]=None):
        if isinstance(data, str) and os.path.isfile(data):
            with open(data, 'r') as f:
                data = f.readlines()
        if isinstance(data, str):
            data = data.split('\n')

        self.data: list[str] = data
        self.name = name if name else self.__class__.__name__
    
    def run(self, model, tokenizer, *args, **kwargs) -> Any:
        '''
            Function which runs on the model with the specified data
        '''
        raise NotImplementedError('Please do not use the baseline class. Implement the method `run` into the subclass.') 
    
    def update_data(self, new_data):
        self.data = new_data


class Metrics:
    def __init__(self, metrics: list[Metric]):
        self.metrics = {
            metric.name: metric
            for metric in metrics
        }

    def run(self, *args, **kwargs):
        results = {}
        for metric_name, metric in self.metrics.items():
            results[metric_name] = metric.run(*args, **kwargs)
        return results
    
    def update_data(self, new_data, metric_id: Optional[str]=None):
        if metric_id is not None:
            if not metric_id in self.metrics.keys(): return False
            return self.metrics[metric_id].update_data(new_data)

        for metric in self.metrics.values():
            metric.update_data(new_data)
        return True