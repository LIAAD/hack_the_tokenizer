# src/__init__.py
from .benchmark import BENCHMARKS
from . import utils, loader, hack, benchmark, metrics
from .BPE import BPE
from .evaluation import evaluation


__all__ = ['utils', 'BENCHMARKS', 'BPE', 'loader', 'evaluation', 'benchmark', 'hack', 'metrics']



__version__ = '0.1.0'