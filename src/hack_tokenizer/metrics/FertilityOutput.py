from typing import Union
from typing_extensions import override

import re

import torch

from .base import Metric


class FertilityOutput(Metric):

    def __init__(self, data: Union[str, list[str]]):
        output = super().__init__(data)
        assert len(self.data) > 1, 'Too little data to calculate fertility. Please pass a bigger dataset'
        return output

    def num_tokens_to_first_full_word(self, tokenizer, outputs: list[int]) -> int:
        """
        Returns the number of tokens needed to generate the first full word,
        skipping partial beginnings (e.g., starting mid-word).
        """
        text = ""
        num_tokens = 0
        total_counter = 0
        word_pattern = re.compile(r'\b\w+\b')  # full word match
        reset_counter = True

        while num_tokens < len(outputs):
            num_tokens += 1
            total_counter += 1
            text = tokenizer.decode(outputs[:num_tokens])
            
            # Find all full words
            words = list(word_pattern.finditer(text))

            if len(words) < 2 and (len(words) == 0 or words[0].start() == 0):
                continue

            if reset_counter:
                num_tokens = 1
                reset_counter = False
            # Check if the first word starts AFTER the beginning (i.e., not starting mid-word)
            iter = words[1:] if words[0].start() == 0 else words
            for _ in iter:
                last_char = text[:_.end()+1][-1]
                if last_char.isalpha() or last_char.isnumeric():  # Make sure this is not an "alphanumeric" character so that we're not in the middle of the sentence
                    continue
                return num_tokens - 1

        return num_tokens

    @override
    def run(self, model, tokenizer, *_, **__):
        fertilities = []
        for text in self.data:
            # text = "Ela correu para chegar a linha de chegada"
            tokenized_words = tokenizer.encode(text)
            fertilities.append({'data': []})
            for n in range(6, len(tokenized_words)+1):  # we start with "6" tokens as it is very unlikely that the model generates something correct with only 1 token
                outputs = model.generate(
                    input_ids=torch.tensor([tokenized_words[:n]], dtype=torch.long, device=model.device),
                    attention_mask=torch.ones((1, n), device=model.device),
                    max_new_tokens=10,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                    **dict(top_p=None, top_k=None, temperature=None, do_sample=False)
                )[0, n:]
                fertilities[-1]['data'].append(self.num_tokens_to_first_full_word(tokenizer, outputs))
            words = re.findall(r'\b\w+\b', text)
            if len(words) == 0: continue
            fertilities[-1]['results'] = sum(fertilities[-1]['data']) / len(fertilities[-1]['data'])

        return sum(fertility['results'] for fertility in fertilities) / len(fertilities)



if __name__ == '__main__':
    # Testing
    import hack_tokenizer.src.utils.loader as loader
    model, tokenizer = loader.load_model_and_tokenizer()
    
    # Load some data
    import hack_tokenizer.src.utils.constants as constants
    with open(constants.DATA_DIR / 'tokenizer_pt-pt.txt', 'r', encoding='utf-8') as f:
        new_tokenizer_dataset = f.readlines()
    # Find new tokens
    from hack_tokenizer.src.hack import ModelHacker
    new_tokens = ModelHacker(dataset=new_tokenizer_dataset, batch_size=-1)._get_new_tokens(
        vocab_size=constants.NUMBER_NEW_TOKENS, ignore_tokens=[tokenizer.decode([x]) for x in range(len(tokenizer))]
    )

    # Filter calamept_dataset to only phrases which include any of the new_tokens
    from hack_tokenizer.src.benchmark.CalamePT import CalamePT
    calamept_dataset_filtered = [text for text in CalamePT().prediction_prompts if any(token in text for token in new_tokens)]
    # pick a random sample
    import random
    random.seed(constants.SEED)
    calamept_dataset_filtered = random.choices(calamept_dataset_filtered, k=2)


    FertilityOutput(calamept_dataset_filtered).run(model, tokenizer)