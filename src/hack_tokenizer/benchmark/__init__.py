from .base import Benchmarks
import os
import pathlib
import importlib

BENCHMARKS = []
for benchmark in os.listdir((pathlib.Path(__file__) / '..').resolve()):
    if benchmark in ['__init__.py', 'base.py'] or not benchmark.endswith('.py'): continue
    # Import the benchmark
    benchmark = benchmark.replace('.py', '')
    bench = importlib.import_module(f'hack_tokenizer.benchmark.{benchmark}')
    BENCHMARKS.append(eval(f'bench.{benchmark}()'))
BENCHMARKS = Benchmarks(BENCHMARKS)
__all__ = ['BENCHMARKS']