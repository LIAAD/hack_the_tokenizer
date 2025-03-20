import torch
import tqdm
from src import utils, loader
DEVICE = 'cpu'

# Load the model and tokenizer
model, tokenizer = loader.load_model_and_tokenizer(device=DEVICE)
hacked_model, hacked_tokenizer = loader.load_model_and_tokenizer(device=DEVICE)

# Phrase that generates " bags"
phrase = 'The airline offers complimentary tags for your checked or carry-on'
original_token_id = model.generate(tokenizer.encode(phrase, return_tensors='pt').to(DEVICE), max_new_tokens=1, pad_token_id=tokenizer.encode('<empty_output>')[0])[0][-1].item()
original_token = tokenizer.decode(original_token_id)    # original_token = " bags", original_token_id=10720

# Add a aditional token to the embedding table (to add a new token)
hacked_model.resize_token_embeddings(len(tokenizer) + 1)

# Get the embedding layer
embedding_layer = hacked_model.get_input_embeddings()

# Weights for the token `original_predicted_token` (' bags')
original_token_embed = embedding_layer(torch.asarray([original_token_id]).to(DEVICE))[0].clone()
new_original_token_embed = torch.zeros(original_token_embed.shape).to(DEVICE)

# Create the new token
new_token = '<new_token>'
new_token_id = len(tokenizer)

# Update embedding table
with torch.no_grad():  # Ensure we modify the tensor without tracking in the computation graph
    embedding_layer.weight[original_token_id].data.copy_(new_original_token_embed)
    embedding_layer.weight[new_token_id].data.copy_(original_token_embed)

assert (hacked_model.get_input_embeddings().weight[new_token_id] == model.get_input_embeddings().weight[original_token_id]).all().item()

# Add a new token to the tokenizer
hacked_tokenizer.add_tokens(new_token)

# Hypothesis 2:
#   So, by default the model does not predict the "new_token_id" because the scores for both 
#       tokens: ' bags' (10720) and ' luggages' (47251) is the same which makes me believe 
#       that the generated token is the first "argmax" (first with the maximum score).
utils.compare_model_generation(
    phrase,
    model, tokenizer,
    hacked_model, hacked_tokenizer,
    max_new_tokens=40,
    DEVICE=DEVICE
)


# --------------------------------------------
# Deleting embedding for the token ' luggages'
# to force the model to pick our token
# --------------------------------------------
inputs = hacked_tokenizer(phrase, return_tensors='pt')
other_token_id = hacked_model.generate(
    inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_new_tokens=1,
    pad_token_id=hacked_tokenizer.eos_token_id
)[-1][-1].item()

# Update embedding table ("zeroing" the embedding for the token generated (which has the same score as our "new_token_id"))
embedding_layer = hacked_model.get_input_embeddings()
with torch.no_grad():
    embedding_layer.weight[other_token_id].data.copy_(new_original_token_embed)

# Generate again
utils.compare_model_generation(
    phrase,
    model, tokenizer,
    hacked_model, hacked_tokenizer,
    max_new_tokens=40,
    DEVICE=DEVICE
)


# --------------------------------------------
# Changing embedding to obtain a better token
# Than our first generated ` bags`
# (basically the embedding ` bags` has a 
# score of 26.3750, we need to find a
# new embedding that produces a better
# result than this.)
# --------------------------------------------

# Iteratively increase each embedding and see how it affects the token generation
new_scores = []
for n in tqdm.trange(original_token_embed.shape[0]):
    new_token_embed = original_token_embed.clone()

    # Increase the value by 40%
    new_token_embed[n] = new_token_embed[n] + new_token_embed[n]*0.4
    with torch.no_grad():
        _ = embedding_layer.weight[new_token_id].data.copy_(new_token_embed)
    # Generate again and store the score
    new_scores.append((
        n, hacked_model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=1
        )['scores'][0][0][new_token_id]
    ))
new_token_embed = original_token_embed.clone()
best_index = max(new_scores, key=lambda x: x[1])[0]
new_token_embed[best_index] += new_token_embed[best_index]*0.4
# Reload the hacked model to swap only new_token_id
hacked_model, hacked_tokenizer = loader.load_model_and_tokenizer(device=DEVICE)
# Add a aditional token to the embedding table (to add a new token)
hacked_model.resize_token_embeddings(len(tokenizer) + 1)
# Update embedding table
embedding_layer = hacked_model.get_input_embeddings()
with torch.no_grad():  # Ensure we modify the tensor without tracking in the computation graph
    _ = embedding_layer.weight[new_token_id].data.copy_(new_token_embed)
# Add a new token to the tokenizer
hacked_tokenizer.add_tokens(new_token)

# Validate generation
utils.compare_model_generation(
    phrase,
    model, tokenizer,
    hacked_model, hacked_tokenizer,
    max_new_tokens=40,
    DEVICE=DEVICE
)


# Validate the generated hidden_layers (trying to understand why it matched with " luggages" and not "<new_token>")
inputs = tokenizer(phrase, return_tensors='pt')
output_original = model.generate(
    inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    pad_token_id=tokenizer.eos_token_id,
    output_scores=True,
    return_dict_in_generate=True,
    max_new_tokens=1
)

output_hacked = hacked_model.generate(
    inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    pad_token_id=tokenizer.eos_token_id,
    output_scores=True,
    return_dict_in_generate=True,
    max_new_tokens=1
)
output_hacked['scores'][0][0][new_token_id]

# Validate that the output of the `new_token_id` == the original output of the `original_token_id`
assert output_hacked['scores'][-1][0][new_token_id] == output_original['scores'][-1][0][original_token_id]

hacked_model.generate()