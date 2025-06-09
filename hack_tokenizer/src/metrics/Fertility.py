from typing import override, Union

from .base import Metric


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
            fertilities.append(len(tokenized_words) / len(text))

        return sum(fertilities) / len(fertilities)



    