from typing import Union
from typing_extensions import override  # ✅ Works in Python 3.10+

import re

from hack_tokenizer.src.metrics.base import Metric


class Fertility(Metric):

    def __init__(self, data: Union[str, list[str]]):
        output = super().__init__(data)
        assert len(self.data) > 10, 'Too little data to calculate fertility. Please pass a bigger dataset'
        return output

    @override
    def run(self, model, tokenizer, *_, **__):
        fertilities = []
        for text in self.data:
            tokenized_words = tokenizer.tokenize(text)
            words = re.findall(r'\b\w+\b', text)
            if len(words) == 0: continue
            fertilities.append(len(tokenized_words) / len(words))

        return sum(fertilities) / len(fertilities)



    