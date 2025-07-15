# src/__init__.py
from .utils.functions import loader, utils
from .benchmark import BENCHMARKS
from . import hack, benchmark, metrics
from .evaluation import evaluation


__all__ = ['utils', 'BENCHMARKS', 'BPE', 'loader', 'evaluation', 'benchmark', 'hack', 'metrics']



__version__ = '0.1.0'