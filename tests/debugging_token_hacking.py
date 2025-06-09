import torch
from hack_tokenizer import loader

DEVICE = 'cuda'
model, tokenizer = loader.load_model_and_tokenizer(device=DEVICE)


# FINDING A PHRASE THAT PREDICTS " luggage" as the new token
phrases = '''
The airline offers complimentary tags for your checked or carry-on  
The conveyor belt at baggage claim was crowded with unclaimed pieces of 
They charge extra for oversized or overweight pieces of checked 
'''
for phrase in phrases[1:-1].split('\n'):
    tokens = tokenizer(phrase.strip(' '), return_tensors='pt')
    tokens = model.generate(tokens['input_ids'].to(DEVICE), max_new_tokens=1, pad_token_id=tokenizer.eos_token_id, attention_mask=tokens['attention_mask'].to(DEVICE))
    print(tokenizer.decode(tokens[0]))



# What happens if I swap an ID that was previously a prediction?
# Example:
phrase = 'The airline offers complimentary tags for your checked or carry-on'
tokens = tokenizer(phrase, return_tensors='pt')
tokens = model.generate(tokens['input_ids'].to(DEVICE), max_new_tokens=1, pad_token_id=tokenizer.eos_token_id, attention_mask=tokens['attention_mask'].to(DEVICE))
token_generated = tokens[0][-1].item()  # token_generated = {ID: 10720, TOKEN: " bags"}

# So, the phrase generated the token `token_generated = 10720`. What happens if I create a new_token with that token_id?
new_token_id = token_generated
new_token = ' martelo'

# ----------------------------
# UPDATING MODEL AND TOKENIZER
original_token_ids = tokenizer.encode(new_token, return_tensors='pt').to(DEVICE)

# Get the embedding layer
embedding_layer = model.get_input_embeddings()
embedding_layer.weight.requires_grad = False  # turn off gradients
# Compute mean embedding of the new tokenizer
mean_emb = embedding_layer(original_token_ids).mean(dim=1)[0]  # (1, 2, 576) -> (batch_size, num_tokens, embedding_dim) ---> (1, 576) -> (batch_size, embedding_dim) ---> (576) -> (embedding_dim)  Drop batch dimension
embedding_layer.weight[new_token_id] = mean_emb
embedding_layer.weight.requires_grad = True  # turn gradients back on
assert torch.allclose(embedding_layer.weight[new_token_id], mean_emb)

# -------------------
# Tokenizer hacking
# -------------------
hacked_tokenizer = utils.replace_tokens(
    tokenizer,
    new_token,
    new_token_id,
    delete_temp_folder=False
)
hacked_tokenizer, hacked_tokenizer_folder = hacked_tokenizer['tokenizer'], hacked_tokenizer['tokenizer_path']


# Now what is the generation of the same phrase that previously predicted Token `token_generated = 47161`?
tokens = hacked_tokenizer(phrase, return_tensors='pt')
tokens = model.generate(tokens['input_ids'].to(DEVICE), max_new_tokens=1, pad_token_id=tokenizer.eos_token_id, attention_mask=tokens['attention_mask'].to(DEVICE))
print(hacked_tokenizer.decode(tokens[0]))

# So, it changed " luggage" to " bags"... This probably means that the output of the model could only find the word " bag" to fit the vector of the output
# This is a good result for us, as this means that whenever we replace a token, the model will try to find the closest token to the token we removed.

