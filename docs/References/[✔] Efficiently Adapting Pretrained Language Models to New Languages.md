# Introduction

This document summarizes the paper ["Efficiently Adapting Pretrained Language Models to New Languages"](https://arxiv.org/pdf/2311.05741).

# Efficiently Adapting Pretrained Language Models to New Languages

## Summary
Paper explores how to efficiently adapt any existing
pretrained LLM to a new language without the need to retrain the model.

Exploring the *Thai* language, tokenizing in Thai results in around *3.8x* more tokens than English[\[1\]](#reference-1).
In addition, the tokenizer is not optimized for the new language, resulting in a sub-optimal tokenization which leads to worse evaluation results[\[2\]](#reference-2).

## Approach
The authors propose to replace the least frequent tokens in the vocabulary of the pretrained LLM with new tokens that are optimized for the new language.

Main idea:
- Train a BPE tokenizer for a new language with a vocabulary of size $k$
- Find number $o$ of overlapping tokens between the pretrained LLM tokens and the new tokenizer vocabulary
- Replace the least $k-o$ non-overlapping tokenizers from the original tokenizer with the newly trained ones
- Calculate the fertility for all of our 3 tokenizers with two test datasets: one containing text in the original tokinzer main language, and another containing text in the new language

## Results
By introducing new tokens, and adding some pre-training [[!!!!! CHECK WHAT THIS PRETRAINING IS, I DIDN'T REALLY FULY UNDERSTAND IT]], the fertility can be drasticly decreased (up to 50%) and the performance of the model can be improved for the new-language while maintaining it's performance for the original language. 


# Conclusion

This paper is a good starting-point to pick up from for this thesis. It shows that it is possible to adapt a pretrained LLM to a new language without the need to retrain the model, by simply replacing the least frequent tokens in the vocabulary of the pretrained LLM with new tokens that are optimized for the new language.

We may explore how we can replace the "replacable" tokens [[!!!!! forgot the name of this type of token, but they are "custom" tokens which do not serve any purpose]].

# References
<span id="reference-1">[1] - EFFICIENT AND EFFECTIVE TEXT ENCODING FOR CHINESE LLAMA AND ALPACA</span><br>
<span id="reference-2">[2] P. Rust, J. Pfeiffer, I. Vulic, S. Ruder, and I. Gurevych, “How good is your tokenizer? on the monolingual performance of multilingual language models,” 2021.</span><br>
<span id="reference-3">[3] F. Stollenwerk, “Training and evaluation of a multilingual tokenizer for gpt-sw3,” 2023.</span><br>
<span id="reference-4">[4] R. Pires, H. Abonizio, T. S. Almeida, and R. Nogueira, “Sabiá: Portuguese large language models,” 2023.</span><br>
<span id="reference-5">[5] J. Ács. (2019, February) Exploring bert’s vocabulary. [Online]. Available: [https://juditacs.github.io/2019/02/19/bert-tokenization-stats.html](https://juditacs.github.io/2019/02/19/bert-tokenization-stats.html)</span><br>
