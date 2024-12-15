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

        config = MODEL_CONFIGS[MODEL] if isinstance(MODEL_CONFIGS[MODEL], dict) else {}
        # Adding device to config if it has not been specified
        config.setdefault('pipeline_kwargs', {})
        config['pipeline_kwargs']['device'] = config['pipeline_kwargs'].get('device', DEVICE)

        # Update config if necessary
        BENCHMARKS.config = config if config != {} else INITIAL_CONFIG.copy()
        BENCHMARKS.run(MODEL, pipeline_kwargs=config['pipeline_kwargs'])

if __name__ == '__main__':
    main()
