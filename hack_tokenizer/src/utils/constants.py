from typing import Callable, Union, Any, Literal, overload
from pathlib import Path
import transformers

TOKENIZER_TYPE = Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]
MODEL_TYPE = transformers.AutoModel
DATA_DIR = Path(__file__).parent.parent.parent.parent / 'data'

SEED = 42


DEVICE                  = 'cuda'
GENERATION_BATCH_SIZE   = 8
MODEL                   = 'HuggingFaceTB/SmolLM3-3B' # 'HuggingFaceTB/SmolLM2-135M' # 'Qwen/Qwen2.5-1.5B-Instruct'
TEMPERATURE             = None
LEARNING_RATE           = 1e-6
NUMBER_NEW_TOKENS       = 5000