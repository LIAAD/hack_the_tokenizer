import tqdm
import re
import torch
import transformers

NEW_TOKEN = 'martelo'
ORIGINAL_PHRASE = 'O meu martelo é forte. Eu gostava que toda a gente tivesse o meu'
DEVICE = 'cuda'

# Load the model and tokenizer
model_name = "HuggingFaceTB/SmolLM-135M" 
model = transformers.AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

#===============================================================================================================
#  1. Original Generation 
#
# Generate 5 new tokens (we only want to retrieve the next "word" and make that a token) [We want this phrase to generate the NEW_TOKEN]
input_tokens = tokenizer.encode(ORIGINAL_PHRASE, return_tensors='pt').to(DEVICE)
token = model.generate(input_tokens, max_new_tokens=2, pad_token_id=tokenizer.eos_token_id)
token = re.split('[ ,.!?;:]', tokenizer.decode(token[0]).replace(ORIGINAL_PHRASE, '').strip())[0]       # token = 'martelo' |token_ids = [39339, 35304] -> ("mart", "elo")
assert token == NEW_TOKEN

# Store the hidden original states in a variable
original_generation = model(input_tokens, max_new_tokens=2)

#===============================================================================================================


#===============================================================================================================
# 2. New Token Embed Calculation
#
# Find out the original token_ids
original_token_ids = tokenizer.encode(token, return_tensors='pt').to(DEVICE)

# Get the embedding layer
embedding_layer = model.get_input_embeddings()
embedding_layer.weight.requires_grad = False  # turn off gradients

# Compute mean embedding of the new tokenizer
ids_embds = embedding_layer(original_token_ids)  # (1, 2, 576) -> (batch_size, num_tokens, embedding_dim)
ids_embds = ids_embds.mean(dim=1)  # (1, 576) -> (batch_size, embedding_dim)
mean_emb = ids_embds[0]  # (576) -> (embedding_dim)  Drop batch dimension
#===============================================================================================================


#===============================================================================================================
# 3. Choosing the ID for the new token
#
new_token_id = embedding_layer._parameters['weight'].shape[0] - 1   # 49
#===============================================================================================================

#===============================================================================================================
# 4. "Hack" both model embedding layer and tokenizer with new token and new id

# ---------------
# Model hacking
# ---------------

# Set id of the embedding layer to the mean embedding
embedding_layer.weight[new_token_id] = mean_emb
embedding_layer.weight.requires_grad = True  # turn gradients back on

assert torch.allclose(embedding_layer.weight[new_token_id], mean_emb)

# Validating the model embedding layer has the new token values
assert (model.get_input_embeddings().weight[new_token_id] == mean_emb).all().item()

# -------------------
# Tokenizer hacking
# -------------------

# Delete previous token with the "new_token_id"
old_token = tokenizer.convert_ids_to_tokens(new_token_id)   # Finding old token
if old_token in tokenizer.vocab:
    tokenizer.vocab.pop(old_token) # Deleting old token from vocab
if old_token in tokenizer.added_tokens_encoder:
    tokenizer.added_tokens_encoder.pop(old_token)   # And from added tokens if it's in there

# Setting up the ID to the new Token
tokenizer.vocab[NEW_TOKEN] = new_token_id
tokenizer.added_tokens_encoder[NEW_TOKEN] = new_token_id
tokenizer.added_tokens_decoder[new_token_id] = NEW_TOKEN
#===============================================================================================================


#===============================================================================================================
# 4. "Hacked" Generation (with new Token)
#
input_tokens = tokenizer.encode(ORIGINAL_PHRASE, return_tensors='pt').to(DEVICE)
raw = model(input_tokens)
tokens = model.generate(input_tokens)

#===============================================================================================================







#----------------- OPTION 2


# 1. Novo token no "New Tokenizer", verificar qual o token que não está no tokenizer "original"
# 2. Depois de arranjar um novo tokenizer, perceber em que posicao coloca-lo (comecar pelo fim)


"""
This script shows how to replace the embedding of a specific token.

As an example, we use the SmolLM-135M model and replace the embeddings at index 123 with the mean embedding of the word "martelo".
"""



input_tokens = tokenizer.encode('o meu martelo é muito forte. O meu ', return_tensors='pt')
raw = model(input_tokens)
tokens = model.generate(input_tokens)

# How do I get from "raw" to "tokens"? ANSWER: raw[0] -> In each index do a soft max and pick the index of the max

print('test')