from hack_tokenizer import loader
import torch

DEVICE = 'cuda'
model, tokenizer = loader.load_model_and_tokenizer(device=DEVICE)

# HYPOTHESIS: The ACTUAL ID of a token doesn't really matter, only it's vector in the embedding table
# How can we test this?
# Let's swap 2 tokens (e.g: " luggage" and " bags") in the embedding layer (meaning we will assign the weights of " bags" to " luggage" and vice-versa)
#   and later validate which token is predicted using the previous input
phrase = 'The airline offers complimentary tags for your checked or carry-on'
token1, token2 = ' bags', ' luggage'
token1_id, token2_id = tokenizer.encode([token1, token2])
input_tokens = tokenizer(phrase, return_tensors='pt')

# Validating the model predicts "token1" as it's initial prediction
original_generated_token = model.generate(input_tokens['input_ids'].to(DEVICE), max_new_tokens=1, pad_token_id=tokenizer.eos_token_id, attention_mask=input_tokens['attention_mask'].to(DEVICE))[-1][-1].item()
assert original_generated_token == token1_id


# SWAPPING both tokens embeds
embedding_layer = model.get_input_embeddings()
# Get the embeddings of our 2 tokens
token1_embed = embedding_layer.weight[token1_id].clone()
token2_embed = embedding_layer.weight[token2_id].clone()
with torch.no_grad():  # Ensure we modify the tensor without tracking in the computation graph
    embedding_layer.weight[token1_id].data.copy_(token2_embed)
    embedding_layer.weight[token2_id].data.copy_(token1_embed)
# Assert the weights have been changed
assert (token1_embed == model.get_input_embeddings().weight[token2_id]).all().item()


# Compare the generation now
# Validating the model predicts "token2" as it's "hacked" prediction
hacked_generated_token = model.generate(input_tokens['input_ids'].to(DEVICE), max_new_tokens=1, pad_token_id=tokenizer.eos_token_id, attention_mask=input_tokens['attention_mask'].to(DEVICE))[-1][-1].item()
assert hacked_generated_token == token2_id

