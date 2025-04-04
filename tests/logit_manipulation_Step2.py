# =======================================================================================================================================
# =======================================================================================================================================
#
#
#                                                               MARK: Step 2
#
#
# =======================================================================================================================================
# =======================================================================================================================================

# Calculating the above steps for all the tokens generated in the dataset we have.
# Later we can check the scores for the benchmarks we have
import torch
import sys
import pathlib
import os
os.chdir('/home/yali/MEGA/Hack The Tockenizer/tests')
sys.path.insert(1, str(pathlib.Path('..').resolve()))
from src import utils, loader, hack
from src.DatasetClass import ListDataset, TextDataset
from torch.utils.data import DataLoader

import tqdm
import transformers
DEVICE = 'cuda'
model_name = 'HuggingFaceTB/SmolLM-135M'
phrases: list[str] = hack.BENCHMARKS.benchmarks[1].prediction_prompts.to_list() # CalamePT dataset 




# Load the model and tokenizer
model, tokenizer = loader.load_model_and_tokenizer(device=DEVICE, tokenizer_kwargs={'padding_side': 'left'})
hacked_model, hacked_tokenizer = loader.load_model_and_tokenizer(device=DEVICE, tokenizer_kwargs={'padding_side': 'left'})
hacked_tokenizer.pad_token_id = hacked_tokenizer.eos_token_id
tokenizer.pad_token_id = tokenizer.eos_token_id

# Step 1. Train a new portuguese vocabulary
pt_tokenizer = hack.TokenizerHack(device=DEVICE).train_tokenizer(trainer_kwargs={'vocab_size': len(tokenizer)})

# Step 2. Find tokens in `pt_tokenizer` not in 
new_tokens = set(pt_tokenizer.get_vocab().keys())
new_tokens = new_tokens.difference(set(tokenizer.vocab.keys()))

# Step 3. Add the new_tokens to model
hacked_model.resize_token_embeddings(len(tokenizer) + len(new_tokens))
hacked_tokenizer.add_tokens(list(new_tokens))

# Step 4. Calculate the new embeddings for the new tokens
embed = model.get_input_embeddings().weight.clone().to('cpu')
new_embed = hacked_model.get_input_embeddings()

# Step 4.1 Initialize everything with averages
# TODO: Change this from AVERAGES to the WEIGHTED average giving more weight to the first tokens
# with torch.no_grad():
#     for new_token in tqdm.tqdm(new_tokens, desc='Initializing the embeddings for the new_tokens'):
#         new_token_id = hacked_tokenizer.encode(new_token)[0]
#         # Find the old embedding for the token
#         tokenization = tokenizer.encode(new_token)
#         token_embed = torch.stack([embed[t_id] for t_id in tokenization]).to(DEVICE)
#         # Create a new token embed using the average
#         new_token_embed = token_embed.mean(dim=0)
#         # Update embedding of the new_token in the hacked_model
#         _ = new_embed.weight[new_token_id].data.copy_(new_token_embed)

# Weighted Average Method
K = 5
with torch.no_grad():
    for new_token in tqdm.tqdm(new_tokens, desc='Initializing the embeddings for the new_tokens'):
        new_token_id = hacked_tokenizer.encode(new_token)[0]
        # Find the old embedding for the token
        tokenization = tokenizer.encode(new_token)
        token_embed = torch.stack([embed[t_id] for t_id in tokenization]).to(DEVICE)
        # Calculating the embedding weights
        embedding_weights = torch.asarray([K**i if K**i < 2**64 else 0 for i in range(token_embed.shape[0], 0, -1)]).to(DEVICE)
        # embedding_weights = torch.asarray([K**i for i in range(token_embed.shape[0], 0, -1)]).to(DEVICE)
        embedding_weights = embedding_weights / embedding_weights.sum()

        # Create a new token embed using the weighted average of the embeddings
        new_token_embed = torch.sum(token_embed * embedding_weights[:, None], dim=0)
        # Update embedding of the new_token in the hacked_model
        _ = new_embed.weight[new_token_id].data.copy_(new_token_embed)


# Step 4.2 Using the training phrases to update the embedding weights
learning_rate = 5e-6
BATCH_SIZE = 64
for new_token in tqdm.tqdm(new_tokens, desc='Updating the embeddings for the new tokens'):
    new_token_id = hacked_tokenizer.convert_tokens_to_ids(new_token)
    new_token = hacked_tokenizer.decode(new_token_id)
    phrases_to_generate_new_token = [p for phrase in phrases for p in phrase.split(new_token) if new_token in phrase and len(p) > 0]

    if len(phrases_to_generate_new_token) == 0: continue
    # Creating the Batched dataset (to run generation for multiple phrases at the same time)
    dataloader = DataLoader(
        TextDataset(phrases_to_generate_new_token, hacked_tokenizer, max_length=max(len(hacked_tokenizer.tokenize(x)) for x in phrases_to_generate_new_token)),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    # Process the batches
    for batch in tqdm.tqdm(dataloader,  desc=f'  Generating tokens for new_token=`{new_token}` ', leave=False):
        # Move batch tensors to the correct device
        input_ids = batch['input_ids'].squeeze(1).to(DEVICE)
        attention_mask = batch['attention_mask'].squeeze(1).to(DEVICE)

        # Generate text
        outputs = hacked_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            num_beams=1,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_logits=True,
            output_scores=True,
            output_hidden_states=True,
            pad_token_id=hacked_tokenizer.pad_token_id
        )

        # Extract the generated sequences and their scores
        generated_sequences = outputs.sequences
        predicted_logits = outputs.logits

        # Decode the input and generated sequences
        input_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        generated_texts = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
        with torch.no_grad():
            for i in range(len(input_texts)):
                # generation.append({
                #     'generated_sequences': generated_sequences[i].to('cpu'),
                #     'prediction_scores': predicted_logits[0][i].to('cpu'),
                #     'input_texts': input_texts[i],
                #     'generated_texts': generated_texts[i],
                #     'hidden_states': [hidden_state[i].to('cpu') for hidden_state in outputs.hidden_states[0]]
                # })

                logits = predicted_logits[0][i]
                logit_gradient = logits.max() - logits[new_token_id]
                embed_out = outputs.hidden_states[0][-1][i][-1]
                embed_in = new_embed.weight[new_token_id]

                # Update the embedding table
                _ = new_embed.weight[new_token_id].data.copy_((embed_in + logit_gradient * embed_out * learning_rate).to(DEVICE))



hacked_tokenizer.decode(
    utils.generate(
        hacked_model,
        hacked_tokenizer,
        phrases[0],
        return_dict_in_generate=False,
        output_logits=False,
        max_new_tokens=10,
        device=DEVICE
    )[0]
)

utils.compare_model_generation(
    phrases[0],
    model, tokenizer,
    hacked_model, hacked_tokenizer,
    max_new_tokens=40,
    DEVICE=DEVICE
)

# Generation with tokenization from original model and the "hacked_model"
phrase = phrases[0]

for _ in range(10):
    inputs = tokenizer(phrase, return_tensors='pt')
    inputs['input_ids'] = inputs['input_ids'].squeeze(1).to(DEVICE)
    inputs['attention_mask'] = inputs['attention_mask'].squeeze(1).to(DEVICE)
    gen = hacked_model.generate(
        **inputs,
        max_new_tokens=1,
        pad_token_id=tokenizer.eos_token_id 
    )
    phrase = hacked_tokenizer.decode(gen[0])
print(phrase)

hacked_tokenizer.tokenize('fug')

# Run the benchmark for the new hacked model to start getting results
hack.BENCHMARKS.config['model_kwargs'] = {'pad_token_id': tokenizer.eos_token_id}
hack.BENCHMARKS.config['parallel_batch_size'] = 32
hack.BENCHMARKS.run(
    hacked_model,
    hacked_tokenizer
)

results = hack.BENCHMARKS.get_results()
import json
with open('/home/yali/MEGA/Hack The Tockenizer/tests/results_hacked.json', 'w') as f:
    json.dump(results, f, indent=2)

# Only thing predicted was a "P"


import json
import pathlib
with open(pathlib.Path(__file__ + '/../results_hacked.json').resolve().as_posix(), 'r') as f:
# with open(r'/mnt/c/Users/yakim/Documents/MEGA/03. Vida Académica/03. Mestrado Ciencias Computadores/Dissertacao/Hack The Tockenizer/tests/results_hacked.json', 'r') as f:
    results = json.load(f)

superglue_predictions = results['HuggingFaceTB/SmolLM-135M']['SupergluePTPT']['results']['HuggingFaceTB/SmolLM-135M']['results-raw']
superglue_predictions = [p['prediction_text'] for i in range(len(superglue_predictions)) for p in superglue_predictions[i]['benchmark_predictions']]
calame_predictions = results['HuggingFaceTB/SmolLM-135M']['CalamePT']['results']['HuggingFaceTB/SmolLM-135M']['results-raw']



# Running the benchmark with the original model
# Run the benchmark for the new hacked model to start getting results
model, tokenizer = loader.load_model_and_tokenizer(device=DEVICE, tokenizer_kwargs={'padding_side': 'left'})
tokenizer.pad_token_id = tokenizer.eos_token_id
hack.BENCHMARKS.config['model_kwargs'] = {'pad_token_id': tokenizer.eos_token_id}
hack.BENCHMARKS.config['parallel_batch_size'] = 32
hack.BENCHMARKS.run(
    model,
    tokenizer
)

results = hack.BENCHMARKS.get_results()
import json
with open('/home/yali/MEGA/Hack The Tockenizer/tests/results_original.json', 'w') as f:
    json.dump(results, f, indent=2)


inputs = tokenizer(hack.BENCHMARKS.benchmarks[0].prediction_prompts[0], return_tensors='pt')
tokenizer.decode(model.generate(
    input_ids=inputs['input_ids'].to(DEVICE),
    attention_mask = inputs['attention_mask'].to(DEVICE),
)[0]
)
    