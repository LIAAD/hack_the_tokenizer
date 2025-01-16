import torch

from src import utils, loader

DEVICE = 'cpu'
NEW_TOKEN = ' preemptive'
ORIGINAL_PHRASE = 'O meu amigo tem um belo martelo'

# Load the model and tokenizer
model, tokenizer = loader.load_model_and_tokenizer(device=DEVICE)
hacked_model, _ = loader.load_model_and_tokenizer(device=DEVICE)

#===============================================================================================================
#                                       MARK: 1. Original Generation 
# Generate 5 new tokens (we only want to retrieve the next "word" and make that a token) [We want this phrase to generate the NEW_TOKEN]
input_tokens = tokenizer(ORIGINAL_PHRASE, return_tensors='pt')
input_tokens, attention_mask = input_tokens['input_ids'].to(DEVICE), input_tokens['attention_mask'].to(DEVICE)
token = model.generate(input_tokens, max_new_tokens=40, pad_token_id=tokenizer.eos_token_id, attention_mask=attention_mask)
print(f"{'':-^100s}\n\n{'Original generation': ^100s}\n\n{'`' + tokenizer.decode(token[0]) + '`': ^100s}\n\n{'':-^100s}")
token = tokenizer.decode(token[0]).replace(ORIGINAL_PHRASE, '')       # token = ' martelo' | token_ids = [11708, 35304] -> ('Ġmart', "elo")
# assert token == NEW_TOKEN # ( Actually the token is 'Ġmart', 'elo' because of the "space" before... So it didn't actually assert correctly, maybe fix it later

# Store the hidden original states in a variable
original_generation = model(input_tokens, max_new_tokens=2)     # 'O meu martelo é forte. Eu gostava que toda a gente tivesse o meu martelo'
#===============================================================================================================


#===============================================================================================================
#                                   MARK: 2. New Token Embed Calculation
# Find out the original token_ids
original_token_ids = tokenizer.encode(token, return_tensors='pt').to(DEVICE)

# Get the embedding layer
embedding_layer = hacked_model.get_input_embeddings()
embedding_layer.weight.requires_grad = False  # turn off gradients

# Compute mean embedding of the new tokenizer
ids_embds = embedding_layer(original_token_ids)  # (1, 2, 576) -> (batch_size, num_tokens, embedding_dim)
ids_embds = ids_embds.mean(dim=1)  # (1, 576) -> (batch_size, embedding_dim)
mean_emb = ids_embds[0]  # (576) -> (embedding_dim)  Drop batch dimension
#===============================================================================================================


#===============================================================================================================
#                           MARK: 3. Choosing the ID for the new token
new_token_id = embedding_layer._parameters['weight'].shape[0] - 1   # 49151
# new_token_id = 49151   # tokenizer.encode(new_token) = [11708, 35304]
#===============================================================================================================


#===============================================================================================================
#           MARK: 4. "Hack" both model embedding layer and tokenizer with new token and new id
# ---------------
# Model hacking
# ---------------
# Set id of the embedding layer to the mean embedding
embedding_layer.weight[new_token_id] = mean_emb
embedding_layer.weight.requires_grad = True  # turn gradients back on

assert torch.allclose(embedding_layer.weight[new_token_id], mean_emb)

# Validating the model embedding layer has the new token values
assert (hacked_model.get_input_embeddings().weight[new_token_id] == mean_emb).all().item()

# -------------------
# Tokenizer hacking
# -------------------
hacked_tokenizer = utils.replace_tokens(
    tokenizer,
    NEW_TOKEN,
    new_token_id,
    delete_temp_folder=False
)
hacked_tokenizer, hacked_tokenizer_folder = hacked_tokenizer['tokenizer'], hacked_tokenizer['tokenizer_path']
#===============================================================================================================


#===============================================================================================================
#                       MARK: 5. "Hacked" Generation (with new Token)
input_tokens_hacked = hacked_tokenizer(ORIGINAL_PHRASE, return_tensors='pt')
input_tokens_hacked, attention_mask_hacked = input_tokens_hacked['input_ids'].to(DEVICE), input_tokens_hacked['attention_mask'].to(DEVICE)
hacked_token_generated = hacked_model.generate(input_tokens_hacked, max_new_tokens=40, pad_token_id=hacked_tokenizer.eos_token_id, attention_mask=attention_mask_hacked)
hacked_generation = hacked_model(input_tokens_hacked, max_new_tokens=5)

print("{:-^100s}\n{: ^100s}\n\n{: ^100s}\n\n{:-^100s}".format(
    '',
    f"Hacked generation | new_token=({new_token_id}, '{NEW_TOKEN}'):", 
    f'`{hacked_tokenizer.decode(hacked_token_generated[0])}`',
    ''
))
#===============================================================================================================



# MARK: Test 1: Swapping two tokens embeddings (validate the input with one token generates the same answer as the hacked with the other swapped token)
import torch
from src import utils, loader
DEVICE = 'cpu'

# Load the model and tokenizer
model, tokenizer = loader.load_model_and_tokenizer(device=DEVICE)
hacked_model, _ = loader.load_model_and_tokenizer(device=DEVICE)

def compare_model_generation(input_phrase, model1, tokenizer1, model2, tokenizer2, max_new_tokens=40):
    
    input_tokens1 = tokenizer1(input_phrase, return_tensors='pt')
    input_tokens1, attention_mask1 = input_tokens1['input_ids'].to(DEVICE), input_tokens1['attention_mask'].to(DEVICE)
    generation1 = model1.generate(input_tokens1, max_new_tokens=max_new_tokens, pad_token_id=tokenizer1.eos_token_id, attention_mask=attention_mask1)
    
    input_tokens2 = tokenizer2(input_phrase, return_tensors='pt')
    input_tokens2, attention_mask2 = input_tokens2['input_ids'].to(DEVICE), input_tokens2['attention_mask'].to(DEVICE)
    generation2 = model2.generate(input_tokens2, max_new_tokens=max_new_tokens, pad_token_id=tokenizer2.eos_token_id, attention_mask=attention_mask2)
    print("{:-^100s}\n{: ^100s}\n\n{: ^100s}\n\n{:-^100s}".format(
        '',
        f"Original Model Generation:", 
        f'`{tokenizer1.decode(generation1[0])}`',
        ''
    ))
    print("{:-^100s}\n{: ^100s}\n\n{: ^100s}\n\n{:-^100s}".format(
        '',
        f"Hacked generation:", 
        f'`{tokenizer2.decode(generation2[0])}`',
        ''
    ))
embedding_layer = model.get_input_embeddings()

# Weights for the token " going"
token1 = ' going'
token1_id = tokenizer.encode(token1)[0] # token1_id = 2045
token1_embed = embedding_layer(torch.asarray(token1_id).to(DEVICE)).clone()

# Weights for the token " responsible"
token2 = ' responsible'
token2_id = tokenizer.encode(token2)[0] # token2_id = 3358
token2_embed = embedding_layer(torch.asarray(token2_id).to(DEVICE)).clone()

# Swapping both embeddings
embedding_layer = hacked_model.get_input_embeddings()
with torch.no_grad():  # Ensure we modify the tensor without tracking in the computation graph
    embedding_layer.weight[token1_id].data.copy_(token2_embed)
    embedding_layer.weight[token2_id].data.copy_(token1_embed)

compare_model_generation(
    'I am being very preemptive. I am not going',
    model, tokenizer,
    hacked_model, tokenizer,
    max_new_tokens=40
)
compare_model_generation(
    'I am being very preemptive. I am not responsible',
    model, tokenizer,
    hacked_model, tokenizer,
    max_new_tokens=40
)
# AS EXPECTED, the hacked model generated the response of the original model for the "other" phrase.


# MARK: Test 2. Will now try to swap the token responsible with weighted average of 0.99*responsible + 0.01*going 
import torch
from src import utils, loader
DEVICE = 'cpu'
# Load the model and tokenizer
model, tokenizer = loader.load_model_and_tokenizer(device='cpu')
hacked_model, _ = loader.load_model_and_tokenizer(device='cpu')

embedding_layer = model.get_input_embeddings()

# Weights for the token " going"
token1 = ' going'
token1_id = tokenizer.encode(token1)[0] # token1_id = 2045
token1_embed = embedding_layer(torch.asarray([token1_id]).to(DEVICE)).clone()

# Weights for the token " responsible"
token2 = ' responsible'
token2_id = tokenizer.encode(token2)[0] # token2_id = 3358
token2_embed = embedding_layer(torch.asarray([token2_id]).to(DEVICE)).clone()

# Swapping both embeddings
# Get the embedding layer
embedding_layer = hacked_model.get_input_embeddings()

# Modify the embedding weights directly for token2 ` responsible`
token1_weight = 0.10
with torch.no_grad():  # Ensure we modify the tensor without tracking in the computation graph
    new_weight = (token1_embed * (1-token1_weight) + token2_embed * token1_weight)[0]
    embedding_layer.weight[token2_id].data.copy_(new_weight)

compare_model_generation(
    'I am being very preemptive. I am not going',
    model, tokenizer,
    hacked_model, tokenizer,
    max_new_tokens=40
)
compare_model_generation(
    'I am being very preemptive. I am not responsible',
    model, tokenizer,
    hacked_model, tokenizer,
    max_new_tokens=40
)