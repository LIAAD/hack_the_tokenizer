# Explore if we can use the generation of the model to retrieve information regarding new tokens

# TODO: Evaluate how different methodologies compare:
#   1. Create new tokens and use the training-dataset to train the model on it
#   2. Validate how the logits for the new_tokens varies according to:
#       2.1 Using a random embedding for each new_token
#       2.2 Using average of the previous tokenization - E(new_token) = [E(t1) + E(t2) + ... + E(tN) / N] where ORIGINAL_Tokenization(new_token) = [t1, t2, ..., tN]
#       2.3 Using a weighted average of the previous tokenization - E(new_token) = [w1*E(1) + ... + wN*E(N)].
#           2.3.1 Try first with w_i = w_{i+1} * K for some different K's
#           2.3.2 Try to use a simple regressor model to find different w_i values using the existing merges
#   3. Compare the logits "RANK" of all different steps as well

import torch
import tqdm
import os
os.chdir(r'Hack The Tockenizer')
from hack_tokenizer import utils, loader, hack, BENCHMARKS
DEVICE = 'cpu'

# Load the model and tokenizer
model, tokenizer = loader.load_model_and_tokenizer(device=DEVICE)

phrase: str = BENCHMARKS.get_benchmark_data('list')[0]
inputs = tokenizer.encode(phrase, return_tensors='pt')
for key in inputs.keys(): inputs[key] = inputs[key].to(DEVICE)

model.generate(
    inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_new_tokens=1,
    pad_token_id=tokenizer.eos_token_id,
    output_scores=True,
    return_dict_in_generate=True,
    return_legacy_cache=True,
    return_hidden_states=True       # FIND THIS ONE to obtain the final "embedding" hidden_state
)