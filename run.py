import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.insert(1, '../')
from src import BENCHMARKS

DEVICE: str = 'cuda'
INITIAL_CONFIG: dict = BENCHMARKS.config.copy()
# Loading Configurations
with open('config.json', 'r') as f:
    MODEL_CONFIGS: dict = json.load(f)

def main():
    for MODEL in MODEL_CONFIGS.keys():
        print('{:-^100s}\n\n{: ^100s}\n\n{:-^100s}'.format('', f'Running benchmarks for model `{MODEL}`', ''))

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(MODEL, use_safetensors=True, **MODEL_CONFIGS[MODEL]['model_kwargs']).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(MODEL)

        # Update config if necessary
        BENCHMARKS.config = MODEL_CONFIGS[MODEL] if isinstance(MODEL_CONFIGS[model], dict) else INITIAL_CONFIG.copy()
        BENCHMARKS.run(model, tokenizer)

        # Freeing up model from GPU memory
        model.to('cpu')
        torch.cuda.empty_cache()
        del(model)

if __name__ == '__main__':
    main()
