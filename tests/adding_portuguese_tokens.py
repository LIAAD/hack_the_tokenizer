import torch
import tqdm
from src import utils, loader, hack
DEVICE = 'cpu'

# Load the model and tokenizer
model, tokenizer = loader.load_model_and_tokenizer(device=DEVICE)
hacked_model, hacked_tokenizer = loader.load_model_and_tokenizer(device=DEVICE)

# Step 1. Train a new portuguese vocabulary
pt_tokenizer = hack.TokenizerHack(device=DEVICE).train_tokenizer(trainer_kwargs={'vocab_size': len(tokenizer)})

# Step 2. Find tokens in `pt_tokenizer` not in 
new_tokens = set(pt_tokenizer.get_vocab().keys())
new_tokens = new_tokens.difference(set(tokenizer.vocab.keys()))

# Step 3. Add the new_tokens to model
hacked_model.resize_token_embeddings(len(tokenizer) + len(new_tokens))
hacked_tokenizer.add_tokens(list(new_tokens))

# Step 4. Calculate the new embeddings for the new tokens (Average)
embed = model.get_input_embeddings().weight.clone().to('cpu')
new_embed = hacked_model.get_input_embeddings()

with torch.no_grad():
    for new_token in tqdm.tqdm(new_tokens):
        token_id = hacked_tokenizer.encode(new_token)[0]
        # Find the old embedding for the token
        tokenization = tokenizer.encode(new_token)
        token_embed = torch.stack([embed[t_id] for t_id in tokenization]).to(DEVICE)
        # Create a new token embed using the average
        new_token_embed = token_embed.mean(dim=0)
        # Update embedding of the new_token in the hacked_model
        _ = new_embed.weight[token_id].data.copy_(new_token_embed)

assert (hacked_model.get_input_embeddings().weight[token_id] == new_token_embed).all()

# Step 5. Compare generations for both models.
phrase = 'O meu meio de transporte favorito é um módulo que é utilizado para criar um programa que receba o nome e o preço de uma pessoa, e mostre o valor'
utils.compare_model_generation(
    phrase,
    model, tokenizer,
    hacked_model, hacked_tokenizer,
    max_new_tokens=40,
    DEVICE=DEVICE
)

# ------------------------------------------------------
# Understanding why fertility of original 
# `tokenizer` seems lower than `hacked_tokenizer`

tmp1='\n'.join([f'`{tokenizer.decoder.decode([t])}` ({t_id})' for t_id, t in zip(tokenizer.encode(phrase), tokenizer.tokenize(phrase))])
tmp2='\n'.join([f'`{hacked_tokenizer.decoder.decode([t])}` ({t_id})' for t_id, t in zip(hacked_tokenizer.encode(phrase), hacked_tokenizer.tokenize(phrase))])

tmp1 = [tokenizer.decoder.decode([t]) for t in tokenizer.tokenize(phrase)]
tmp2 = [hacked_tokenizer.decoder.decode([t]) for t in hacked_tokenizer.tokenize(phrase)]
tmp = []
c1, c2 = 0, 0
while c1 < len(tmp1) and c2 < len(tmp2):
    len1, len2 = len(''.join(tmp1[:c1])),  len(''.join(tmp2[:c2]))
    if len1 > len2:
        c2 += 1
        tmp[-1][-1] = tmp2[c2]
    elif len1 < len2: 
        c1 += 1
        tmp[-1][0] = tmp1[c1]
    else:
        tmp.append([tmp1[c1], tmp2[c2]])
        c1 += 1
        c2 += 1


# Let's count how many times the `tokenizer` has a fertility lower than `hacked_tokenizer` in the "BENCHMARK" dataset
counter = 0
data = hack.TokenizerHack().train_data
for text in tqdm.tqdm(data):
    if len(tokenizer.tokenize(text)) < len(hacked_tokenizer.encode(text)):
        counter += 1
print(f'# Fertility(`tokenizer`) < Fertility(`hacked_tokenizer`) = {counter / len(data) * 100 :.2f}% ({counter}/{len(data)})')

len(tokenizer.tokenize('\n'.join(data)))
len(hacked_tokenizer.tokenize('\n'.join(data)))
# 2025.01.26 - New findings: the " " token is actually from the original `tokenizer`, not the hacked one.
#   Comparing the fertility for the entire "training data" for the 2 tokenizers, `hacked_tokenizer` is lower, but not by much (1% less (4.171.393 VS 4.202.417))
#   Hypothesis: Adding new tokens to the tokenizer is not as beneficial as to swapping out already existing tokens
#       How to validate the Hypothesis????



# ---------------------
# 2025.01.28 - Exploring bigger models
import torch
import tqdm
import os
os.chdir(r'Hack The Tockenizer')
from src import utils, loader, hack
DEVICE = 'cuda' # Exploring them in CUDA to speed up generation (since they're bigger)
model_name = "HuggingFaceTB/SmolLM-1.7B"

# Only 1 model at a time since this one is bigger
model, tokenizer = loader.load_model_and_tokenizer(model_name, DEVICE)
hacked_tokenizer = loader.load_model_and_tokenizer(model_name, 'cpu')[1] # Only loading the tokenizer since model is too big to fit 2 in memory

phrase = 'O meu meio de transporte favorito é um módulo que é utilizado para criar um programa que receba o nome e o preço de uma pessoa, e mostre o valor'

# Step 1. Find the generation for the original model
inputs = {k: i.to(DEVICE) for k, i in tokenizer(phrase, return_tensors='pt').items()}
original_generation = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], pad_token_id=tokenizer.eos_token_id)

# Step 1.1. Craete the "hacked_model" from original model (no need to load it again)
hacked_model = model

# Step 2. Train a new portuguese vocabulary
pt_tokenizer = hack.TokenizerHack(device=DEVICE).train_tokenizer(trainer_kwargs={'vocab_size': len(tokenizer)})

# Step 3. Find tokens in `pt_tokenizer` not in old tokenizer
new_tokens = set(pt_tokenizer.get_vocab().keys())
new_tokens = new_tokens.difference(set(tokenizer.vocab.keys()))

# Step 4. Add the new_tokens to model
hacked_model.resize_token_embeddings(len(tokenizer) + len(new_tokens))
hacked_tokenizer.add_tokens(list(new_tokens))

# Step 5. Calculate the new embeddings for the new tokens (Average)
embed = model.get_input_embeddings().weight.clone().to('cpu')
new_embed = hacked_model.get_input_embeddings()

with torch.no_grad():
    for new_token in tqdm.tqdm(new_tokens):
        token_id = hacked_tokenizer.encode(new_token)[0]
        # Find the old embedding for the token
        tokenization = tokenizer.encode(new_token)
        token_embed = torch.stack([embed[t_id] for t_id in tokenization]).to(DEVICE)
        # Create a new token embed using the average
        new_token_embed = token_embed.mean(dim=0)
        # Update embedding of the new_token in the hacked_model
        _ = new_embed.weight[token_id].data.copy_(new_token_embed)

# Step 6. Generation for the `hacked_model`
inputs = {k: i.to(DEVICE) for k, i in hacked_tokenizer(phrase, return_tensors='pt').items()}
hacked_generation = hacked_model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], pad_token_id=tokenizer.eos_token_id)

# Step 7. Compare both generations
print("{:-^100s}\n{: ^100s}\n\n{: ^100s}\n\n{:-^100s}".format(
    '',
    f"Original Model Generation:", 
    f'`{tokenizer.decode(original_generation[0])}`',
    ''
))
print("{:-^100s}\n{: ^100s}\n\n{: ^100s}\n\n{:-^100s}".format(
    '',
    f"Hacked generation:", 
    f'`{hacked_tokenizer.decode(hacked_generation[0])}`',
    ''
))



# ------------------------------------------------------
# MARK: Last update
# Adding a simple token which is the merge of 2 others
import numpy as np
import torch
import tqdm
import os
os.chdir(r'Hack The Tockenizer')
from src import utils, loader, hack
DEVICE = 'cpu'

# Load the model and tokenizer
model, tokenizer = loader.load_model_and_tokenizer(device=DEVICE)
hacked_model, hacked_tokenizer = loader.load_model_and_tokenizer(device=DEVICE)

phrase = 'Se um dia alguém, perguntar por'

# Step 1. Finding the 2 generated tokens
inputs = tokenizer(phrase, return_tensors='pt')
new_token = model.generate(inputs['input_ids'].to(DEVICE), attention_mask=inputs['attention_mask'].to(DEVICE), max_new_tokens=2, pad_token_id=tokenizer.eos_token_id)[0][-2:]
new_token = tokenizer.decode(new_token)

# Step 2. Add the new_tokens to model
hacked_model.resize_token_embeddings(len(tokenizer) + 1, mean_resizing=False)
hacked_tokenizer.add_tokens(new_token)

# Step 5. Calculate the new embeddings for the new tokens (Average)
embed = model.get_input_embeddings().weight.clone().to('cpu')
new_embed = hacked_model.get_input_embeddings()

with torch.no_grad():
    token_id = hacked_tokenizer.encode(new_token)[0]
    # Find the old embedding for the token
    tokenization = tokenizer.encode(new_token)
    token_embed = torch.stack([embed[t_id] for t_id in tokenization]).to(DEVICE)
    # Create a new token embed using the average
    new_token_embed = token_embed.mean(dim=0)
    # Update embedding of the new_token in the hacked_model
    _ = new_embed.weight[token_id].data.copy_(new_token_embed)

# Step 6. Verify which token is generated with the hacked model
inputs = hacked_tokenizer(phrase, return_tensors='pt')
new_generation = hacked_model.generate(inputs['input_ids'].to(DEVICE), attention_mask=inputs['attention_mask'].to(DEVICE), max_new_tokens=2, pad_token_id=hacked_tokenizer.eos_token_id)[0]
new_generation = hacked_tokenizer.decode(new_generation)

# MARK: Conclusions
# So the model still generates the old tokens instead of the new one... Let's check the scores for each token to understand why that is
generation = hacked_model.generate(
    inputs['input_ids'].to(DEVICE),
    attention_mask=inputs['attention_mask'].to(DEVICE),
    max_new_tokens=1,
    pad_token_id=hacked_tokenizer.eos_token_id,
    output_scores=True,
    return_dict_in_generate=True,
    return_legacy_cache=True,

)

scores = np.array(generation['scores'][0][0].tolist())
# Comparing new token score and chosen token score
print(f'New Token Score: {scores[-1]} || Old Token Score: {scores.max()}')    # New Token Score: 2.421875 || Old Token Score: 7.09375
# The difference is very high: This is because why would the model select the second token in the first generation?
# I believe we'll have to train the model quite a bit...

# Checking the scores of the two tokens " u" and "ma"
print(f"Score ` u` = {scores[tokenizer.encode(' u')[0]]} | Score `ma` = {scores[tokenizer.encode('ma')[0]]} | Score  ` uma` = {scores[-1]}")
print(f"Avg Score ` u` + `ma` = {(scores[tokenizer.encode(' u')[0]] + scores[tokenizer.encode('ma')[0]]) / 2}")


# MARK: Hypothesis
# Hypothesis: The score of the mean of the embed, is ~ the mean of the scores

def get_nth_max(arr, nth, option=1):
    if option==0:
        val, index = sorted([ (x,i) for i,x in enumerate(arr)], reverse=True)[nth-1]
    else:
        index = np.argpartition(arr, -nth)[-nth]
        val = arr[index]
    return {'value': val, 'index': int(index)}

scores_sorted = sorted([ (x,i) for i,x in enumerate(scores)], reverse=True)
[(nth, x) for nth, x in enumerate(scores_sorted) if x[-1] == hacked_tokenizer.encode(' uma')[0]]
get_nth_max(scores_sorted, 200, option=0)

# MARK: Second approach
# Trying to check how the model generates just a single "new_token"
# My idea is: The model was not trained with the new embeddings we are creating, so it gets confused...
# I will choose a new token from the new_tokenizer and understand how the model compares with the hacked model with the same input text
import numpy as np
import torch
import tqdm
import os
os.chdir(r'Hack The Tockenizer')
from src import utils, loader, hack
DEVICE = 'cpu'


# Load the model and tokenizer
model, tokenizer = loader.load_model_and_tokenizer(device=DEVICE)
hacked_model, hacked_tokenizer = loader.load_model_and_tokenizer(device=DEVICE)

# Trying to use "ajuda" as a new token to see how it is seen by the model
new_token = 'O meu português é'
hacked_tokenizer.add_tokens(new_token)
hacked_model.resize_token_embeddings(len(hacked_tokenizer))
# Asserting the changes were correct and that the "new_token" is not a token from the original tokenizer
assert len(tokenizer.tokenize(new_token)) > 1 and len(hacked_tokenizer.tokenize(new_token)) == 1

# Update the weight for the new_token using the average of the embedding of the original tokenization
embed = model.get_input_embeddings().weight.clone().to('cpu')
new_embed = hacked_model.get_input_embeddings()

with torch.no_grad():
    token_id = hacked_tokenizer.encode(new_token)[0]
    # Find the old embedding for the token
    tokenization = tokenizer.encode(new_token)
    token_embed = torch.stack([embed[t_id] for t_id in tokenization]).to(DEVICE)
    # Create a new token embed using the average
    new_token_embed = token_embed.mean(dim=0)
    # Update embedding of the new_token in the hacked_model
    _ = new_embed.weight[token_id].data.copy_(new_token_embed)

# Compare the generations of the first 40 tokens
utils.compare_model_generation(
    new_token,
    model, tokenizer,
    hacked_model, hacked_tokenizer,
    max_new_tokens=40,
    DEVICE=DEVICE
)
# I believe the problem is because the embedding can't capture the attention between the tokens, 
# how can we have a single embedding, with no attention, generating portuguese text?

# MARK: Test 1: Trying to generate with a few additional tokens to make the text portuguese (how many tokens are needed?)
input_tokens = tokenizer(new_token, return_tensors='pt')
next_tokens = model.generate(input_tokens['input_ids'], attention_mask=input_tokens['attention_mask'], pad_token_id=tokenizer.eos_token_id)[0]
# Delete the input phrase from the next_tokens
next_tokens = next_tokens[input_tokens['input_ids'].shape[1]:]

generations = []
for i in tqdm.tqdm(range(1, len(next_tokens)+1), desc='Generating for different inputs'):
    input_tokens = hacked_tokenizer.encode(new_token, return_tensors='pt')
    # Adding one token at a time to the input_ids
    input_tokens = torch.cat((input_tokens.reshape(input_tokens.shape[1]), next_tokens[:i])).to(DEVICE)
    attention_mask = torch.tensor([[1 for _ in range(len(input_tokens))]]).to(DEVICE)
    generations.append(hacked_model.generate(input_tokens.reshape((1, input_tokens.shape[0])), attention_mask=attention_mask, pad_token_id=hacked_tokenizer.eos_token_id))
    generations[-1] = {'input': f'[{new_token}]' + tokenizer.decode(input_tokens), 'generation': hacked_tokenizer.decode(generations[-1][0])}
for i in generations:
    print(i)
# Visually found that adding 2 tokens (" um", " m") makes the text portuguese:
utils.compare_model_generation(
    new_token + ' um m',
    model, tokenizer,
    hacked_model, hacked_tokenizer,
    max_new_tokens=40,
    DEVICE=DEVICE
)


# MARK: Test 2: Using a different "embedding" calculation to go from "original" to "new"
# This approach will be using a weighted approach, giving more weight to the first tokens
#   Assuming our Tokenization(NEW_TOKEN) = [t1, t2, t3, ..., tn], the Embedding(NEW_TOKEN) = [w1*E(t1) + w2*E(t2) + ...] where
#    w_i = w_{i+1} * X for all i for some X
relative_weight = 5   # X value

# Reloadding the hacked model and tokenizer
hacked_model_test2, hacked_tokenizer_test2 = loader.load_model_and_tokenizer(device=DEVICE)

# Trying to use "ajuda" as a new token to see how it is seen by the model
new_token = 'O meu português é'
hacked_tokenizer_test2.add_tokens(new_token)
hacked_model_test2.resize_token_embeddings(len(hacked_tokenizer_test2))
# Asserting the changes were correct and that the "new_token" is not a token from the original tokenizer
assert len(tokenizer.tokenize(new_token)) > 1 and len(hacked_tokenizer_test2.tokenize(new_token)) == 1

# Update the weight for the new_token using the average of the embedding of the original tokenization
embed = model.get_input_embeddings().weight.clone().to('cpu')
new_embed = hacked_model_test2.get_input_embeddings()

with torch.no_grad():
    token_id = hacked_tokenizer_test2.encode(new_token)[0]
    # Find the old embedding for the token
    tokenization = tokenizer.encode(new_token)
    token_embed = torch.stack([embed[t_id] for t_id in tokenization]).to(DEVICE)
    # Calculating the embedding weights
    embedding_weights = torch.asarray([relative_weight**i for i in range(token_embed.shape[0], 0, -1)])
    embedding_weights = embedding_weights / embedding_weights.sum()

    # Create a new token embed using the weighted average of the embeddings
    new_token_embed = torch.sum(token_embed * embedding_weights[:, None], dim=0)
    # Update embedding of the new_token in the hacked_model_test2
    _ = new_embed.weight[token_id].data.copy_(new_token_embed)
# Understand which changes this does to our generation
utils.compare_model_generation(
    # new_token + ' um m',
    new_token,
    model, tokenizer,
    hacked_model, hacked_tokenizer,
    hacked_model_test2, hacked_tokenizer_test2,
    max_new_tokens=40,
    DEVICE=DEVICE
)

# Hypothesis: To obtain some results we need to replace the tokens from the original tokenizer (due to the model having been trained on the specific tokenizers, it will most likely
# have a bias towards those tokens rather than new ones (it is very unlikely that the model gives a high score to the new tokens))

# MARK: New Tokens Test
# Does the model with the new tokenizers EVER predict a new token? If so, how does it happen?
import torch
import tqdm
import os
os.chdir(r'Hack The Tockenizer')
from src import utils, loader, hack
DEVICE = 'cpu'

# Load the model and tokenizer
_, tokenizer = loader.load_model_and_tokenizer(device=DEVICE)
m, t = loader.load_model_and_tokenizer(device=DEVICE)

# Step 1. Train a new portuguese vocabulary
pt_tokenizer = hack.TokenizerHack(device=DEVICE).train_tokenizer(trainer_kwargs={'vocab_size': len(tokenizer)})

# Step 2. Find tokens in `pt_tokenizer` not in 
new_tokens = set(pt_tokenizer.get_vocab().keys())
new_tokens = new_tokens.difference(set(tokenizer.vocab.keys()))

# Step 3. Add the new_tokens to model
m.resize_token_embeddings(len(t) + len(new_tokens))
t.add_tokens(list(new_tokens))

# Step 4. Calculate the new embeddings for the new tokens (Average)
embed = m.get_input_embeddings().weight.clone().to('cpu')
new_embed = m.get_input_embeddings()

with torch.no_grad():
    for new_token in tqdm.tqdm(new_tokens):
        token_id = t.encode(new_token)[0]
        # Find the old embedding for the token
        tokenization = tokenizer.encode(new_token)
        token_embed = torch.stack([embed[t_id] for t_id in tokenization]).to(DEVICE)
        # Create a new token embed using the average
        new_token_embed = token_embed.mean(dim=0)
        # Update embedding of the new_token in the hacked_model
        _ = new_embed.weight[token_id].data.copy_(new_token_embed)

assert (m.get_input_embeddings().weight[token_id] == new_token_embed).all()

# Step 5. Find if the hacked_model generates a "added_token"
phrase = 'O meu meio de transporte favorito é um módulo que é utilizado para criar um programa que receba o nome e o preço de uma pessoa, e mostre o valor'
input_tokens = t(phrase, return_tensors='pt')
for i in tqdm.trange(1, len(input_tokens['input_ids'][0])):
    tokens = input_tokens['input_ids'][:i].to(DEVICE)
    mask = input_tokens['attention_mask'][:i].to(DEVICE)
    gen = m.generate(tokens, attention_mask=mask, pad_token_id=t.eos_token_id)[0][len(tokens):]
    # Finding if any of the generated tokens is any of the added tokens
    if (gen >= len(tokenizer)).any():
        print('The input `{}` generates the token `{}` ({})'.format(
            t.decode(tokens[0]),
            t.decode(gen[gen >= len(tokenizer)][0]),
            gen[gen >= len(tokenizer)][0]
        ))
        break
# As expected, no "added_token" was generated
# So, we probably do have to replace tokens from the original tokenizer
