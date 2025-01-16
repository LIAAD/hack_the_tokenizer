import transformers
import torch

def load_model_and_tokenizer(
    model_name = "HuggingFaceTB/SmolLM-135M",
    device = 'cuda',
    model_kwargs = {
        'torch_dtype': torch.bfloat16
    },
    tokenizer_kwargs={}
):
    # Load the model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    return model, tokenizer

if __name__ == '__main__':
    DEVICE = 'cuda'
    model, tokenizer = load_model_and_tokenizer(device=DEVICE)