from typing import override, Union

import numpy as np
import torch

from .base import Metric


class Perplexity(Metric):

    def __init__(self, data: Union[str, list[str]]):
        return super().__init__(data)

    @override
    def run(self, model, tokenizer, *_, **__):
        # Pick random 100 samples to calculate perplexity
        random_sample = np.random.choice(self.data, size=100)
        perplexities = []
        for text in random_sample:
            encodings = tokenizer(text, return_tensors='pt').to(model.device)
            
            # Get the input IDs and create attention mask
            input_ids = encodings.input_ids
            attention_mask = encodings.attention_mask
            
            # Calculate perplexity
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                
            # Perplexity = exp(loss)
            perplexities.append(torch.exp(loss).item())
        return sum(perplexities) / len(perplexities)


    