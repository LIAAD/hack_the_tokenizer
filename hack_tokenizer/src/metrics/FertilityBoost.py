from typing import Union
from typing_extensions import override
import tqdm

import torch

from hack_tokenizer.src.metrics.base import Metric


class FertilityBoost(Metric):
    '''
        This metric calculates how many times tokens added are generated (meaning tokens with `token_id` > threshold)
    '''

    def __init__(self, data: Union[str, list[str]]):
        output = super().__init__(data)
        assert len(self.data) > 1, 'Too little data to calculate fertility. Please pass a bigger dataset'
        return output

    @override
    def run(self, model, encode_tokenizer, *_, model_gen_kwargs: dict|None=None, show_progress:bool=False, **__):
        # Threshold for the counting of tokens generated
        threshold = len(encode_tokenizer)
        model_gen_kwargs = model_gen_kwargs if model_gen_kwargs is not None else dict(top_p=None, top_k=None, temperature=None, do_sample=False)
        temperature = model_gen_kwargs.get('temperature')
        boosts = []
        if show_progress: iter_ = tqdm.tqdm(self.data, desc='Calculating `FertilityBoost` Metric')
        else: iter_ = self.data
        for text in iter_:
            # text = "Ela correu para chegar a linha de chegada"
            input_ids = encode_tokenizer.encode(text)
            output = model(
                input_ids=torch.tensor([input_ids], device=model.device),
                max_new_tokens=1,
                **model_gen_kwargs
            )
            if temperature is None:
                predicted_tokens = output.logits.argmax(dim=2)
            else:
                # Sampling of logits when temperature exists
                # Scale by temperature
                scaled_logits = output.logits / temperature
                probs = torch.nn.functional.softmax(scaled_logits, dim=-1).to('cpu')    # Don't have enough memory
                # Flatten for multinomial sampling
                sampled = torch.multinomial(
                    probs.view(-1, probs.size(-1)), num_samples=1
                )
                predicted_tokens = sampled.view(output.logits.size(0), output.logits.size(1)).to('cpu')
                # Release memory from GPU
                del(sampled)
                del(probs)
                del(scaled_logits)
                del(output)
            num_tokens_after_threshold = (predicted_tokens[0] > threshold).sum().item()
            boosts.append(num_tokens_after_threshold / predicted_tokens.size(1))
        return sum(boosts) / len(boosts)



if __name__ == '__main__':
    # Testing
    import hack_tokenizer.src.utils.loader as loader
    import transformers
    model, tokenizer = loader.load_model_and_tokenizer(tokenizer_kwargs={'padding_side': 'left'})
    encoding_tokenizer = transformers.AutoTokenizer.from_pretrained(model.name_or_path, padding_side='left')

    # Load some data
    import hack_tokenizer.src.utils.constants as constants
    with open(constants.DATA_DIR / 'tokenizer_pt-pt.txt', 'r', encoding='utf-8') as f:
        new_tokenizer_dataset = f.readlines()
    
    # Hack model
    from hack_tokenizer.src.hack import ModelHacker
    hacker = ModelHacker(
        dataset=new_tokenizer_dataset,
        batch_size=constants.GENERATION_BATCH_SIZE,
        learning_rate=constants.LEARNING_RATE
    )
    with open('./data/calamept_dataset.txt', 'r', encoding='utf-8') as f:
        train_dataset = f.readlines()
    model, tokenizer = hacker.hack(
        model, tokenizer,
        encoding_tokenizer,
        num_tokens=constants.NUMBER_NEW_TOKENS,
        embed_initializer_method='weighted_drop(1.5)',
        show_progress=True,
        train=False,
        train_kwargs={
            'dataset': train_dataset
        }
    )

    # Find new tokens
    new_tokens = hacker.new_tokens

    # Filter fertility_boosat to only phrases which include any of the new_tokens
    with open(constants.DATA_DIR / 'fertility_boost_evaluation-dataset.txt', 'r') as f:
        dataset = f.readlines()
    dataset_filtered = [text for text in dataset if any(token in text for token in new_tokens)]
    # pick a random sample
    import random
    random.seed(constants.SEED)
    dataset_filtered = random.choices(dataset_filtered, k=1000)

    torch.manual_seed(constants.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(constants.SEED)

    fertility_boost = FertilityBoost(dataset_filtered).run(model, encoding_tokenizer, show_progress=True, model_gen_kwargs={'temperature': 0.8})
    print('Fertility Boost for trained = {:.2%}'.format(fertility_boost))