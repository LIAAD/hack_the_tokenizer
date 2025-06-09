import transformers
import transformers.models as models
import torch
from typing import Tuple

def load_model_and_tokenizer(
    model_name = "HuggingFaceTB/SmolLM-135M",
    # model_name = "HuggingFaceTB/SmolLM-1.7B",
    device = 'cuda',
    model_kwargs = {
        'torch_dtype': torch.bfloat16
    },
    tokenizer_kwargs={}
) -> Tuple[models.llama.modeling_llama.LlamaForCausalLM, models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast]:  # NOTE: Change the type hint depending on the model_name
    # Load the model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    return model, tokenizer

if __name__ == '__main__':
    DEVICE = 'cuda'
    model, tokenizer = load_model_and_tokenizer(device=DEVICE)