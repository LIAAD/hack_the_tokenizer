from .Fertility import Fertility
from .Perplexity import Perplexity
from .base import Metrics
from ..benchmark import BENCHMARKS

_data = BENCHMARKS.get_benchmark_data('list')
METRICS = Metrics([Fertility(_data), Perplexity(_data)])

__all__ = ['Fertility', 'Perplexity', 'METRICS']
