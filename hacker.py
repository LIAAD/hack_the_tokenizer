import torch
from transformers import AutoModelForCausalLM

from src.hack import TokenizerHack
from src import loader
    
if __name__ == '__main__':
    MODEL = 'HuggingFaceTB/SmolLM2-135M'
    DEVICE = 'cuda'

    model, tokenizer = loader.load_model_and_tokenizer(MODEL, DEVICE)
    hack = TokenizerHack(device=DEVICE, model=model, tokenizer=tokenizer)

    # Setting input to test with OLD vs HACKED model
    test_text = 'A gravidade é uma força'
    max_new_tokens = 30

# =================================================================================================
#                                       Original generation
    print(f'{"":-^50s}\n\n{"Testing with Original Model": ^50s}\n\n{"":-^50s}\n')
    input_tokens = tokenizer([test_text], return_tensors='pt')
    model_output = model.generate(
        input_tokens['input_ids'].to(DEVICE), attention_mask=input_tokens['attention_mask'].to(DEVICE),
        max_new_tokens = max_new_tokens,
        pad_token_id = tokenizer.eos_token_id,
        eos_token_id = tokenizer.eos_token_id
    )[0]
    print(tokenizer.decode(model_output))
# =================================================================================================


# =================================================================================================
#                           "Hacking" tokenizer and model + Generation
    print(f'{"":-^50s}\n\n{"Hacking tokenizer & model": ^50s}\n\n{"":-^50s}\n')
    hacked_model = AutoModelForCausalLM.from_pretrained(MODEL, use_safetensors=True, torch_dtype=torch.bfloat16).to(DEVICE)
    hacked = hack.hack(hacked_model, tokenizer)
    
    print(f'{"":-^50s}\n\n{"Testing with hacked tokenizer & model": ^50s}\n\n{"":-^50s}\n')
    input_tokens = hacked['tokenizer'].encode(test_text)
    input_tokens = {
        'input_ids': torch.tensor([input_tokens.ids]),
        'attention_mask': torch.tensor([input_tokens.attention_mask])
    }
    model_output = hacked['model'].generate(
        input_tokens['input_ids'].to(DEVICE), attention_mask=input_tokens['attention_mask'].to(DEVICE),
        max_new_tokens=max_new_tokens,
        pad_token_id = tokenizer.eos_token_id,
        eos_token_id = tokenizer.eos_token_id
    )[0]
    print(hacked['tokenizer'].decode(model_output.tolist()))
# =================================================================================================
