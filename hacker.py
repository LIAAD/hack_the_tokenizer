import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.hack import TokenizerHack
    
if __name__ == '__main__':
    MODEL = 'HuggingFaceTB/SmolLM2-135M'
    DEVICE = 'cuda'

    hack = TokenizerHack(device=DEVICE)
    model = AutoModelForCausalLM.from_pretrained(MODEL, use_safetensors=True, torch_dtype= torch.bfloat16).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Setting input to test with OLD vs HACKED model
    test_text = 'A gravidade é uma força'
    max_new_tokens = 30

    print(f'{"":-^50s}\n\n{"Testing with Original Model": ^50s}\n\n{"":-^50s}\n')
    input_tokens = tokenizer([test_text], return_tensors='pt')
    model_output = model.generate(
        input_tokens['input_ids'].to(DEVICE), attention_mask=input_tokens['attention_mask'].to(DEVICE),
        max_new_tokens = max_new_tokens,
        pad_token_id = tokenizer.eos_token_id,
        eos_token_id = tokenizer.eos_token_id
    )[0]
    print(tokenizer.decode(model_output))
    
    print(f'{"":-^50s}\n\n{"Hacking tokenizer & model": ^50s}\n\n{"":-^50s}\n')
    hacked_model = AutoModelForCausalLM.from_pretrained(MODEL, use_safetensors=True, torch_dtype= torch.bfloat16).to(DEVICE)
    hacked = hack.hack(hacked_model, tokenizer)


#-----------------------
#  DEBUGGING

    # Showing that the codification of "cão" is the average of the hacked one
    test_input = "cão"

    # Encoding with both Hacked and Original tokeni zer
    hacked_input = hacked['tokenizer'].encode(test_input).ids
    original_input = tokenizer(test_input)['input_ids']

    # Caculating embeddings of new and old tokenizer
    hacked_embedding_output = [hacked['model'].get_input_embeddings()._parameters['weight'][token] for token in hacked_input]
    original_embedding_output = [model.get_input_embeddings()._parameters['weight'][token] for token in original_input]

    (original_embedding_output[0][0] + original_embedding_output[1][0]) / 2

    print("Finished debugging")
#  END OF DEBUGGING
#-----------------------

    
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
