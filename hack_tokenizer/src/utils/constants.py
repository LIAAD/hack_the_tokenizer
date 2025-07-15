from typing import Callable, Union, Any, Literal, overload
import transformers

TOKENIZER_TYPE = Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]
MODEL_TYPE = transformers.AutoModel

SEED = 42


DEVICE                  = 'cuda'
GENERATION_BATCH_SIZE   = 8
# MODEL                   = 'Qwen/Qwen2.5-1.5B-Instruct'
MODEL                   = 'HuggingFaceTB/SmolLM2-135M'
TEMPERATURE             = None
LEARNING_RATE           = 1e-6
NUMBER_NEW_TOKENS       = 1000