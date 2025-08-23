from .FertilityInput import FertilityInput
from .FertilityOutput import FertilityOutput
from .FertilityBoost import FertilityBoost
from .Perplexity import Perplexity
from .base import Metrics
from ..benchmark import BENCHMARKS

_data = BENCHMARKS.get_benchmark_data('list')
METRICS = Metrics([FertilityInput(_data), Perplexity(_data), FertilityOutput(_data), FertilityBoost(_data)])

__all__ = ['FertilityInput', 'Perplexity', 'FertilityOutput', 'FertilityBoost', 'METRICS']
