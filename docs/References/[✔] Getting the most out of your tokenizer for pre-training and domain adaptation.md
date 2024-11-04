# Introduction

This document summarizes the paper [Getting the most out of your tokenizer for pre-training and domain adaptation](https://arxiv.org/pdf/2402.01035).

# Getting the most out of your tokenizer for pre-training and domain adaptation

## Summary
Paper explores how the tokenizer impacts the performance. It doesn't focus with an end-goal in mind, just how changing the tokenizer can impact the model.


## Approach
1. Compare pupular code tokenizers, clarifying their respective performances and trade-offs.
1. Study of the impact of vocabulary size, regular expressions for compression on overall model performance


## Results
1. BPE compresses tokenizers quite effectively (40% more than compared to LLama Model, 25% more than GPT4)
1. When adapting a model with new tokenizers, it may be necessary a long fine-tuning (>50B tokens) for training step.
1. Pre-tokenization regular expression is overall a good compression and performance choice.

# Conclusion
This paper can help us determine which methods to use when adding tokenizers to models. 
More specifically, helping in answering:
1. What are the best tokenizers to use for adaptation/extension of vocab?
1. What is the best vocabulary size to extend an initial model with?

# References
<span id="reference-1">[1] - </span><br>
<span id="reference-2">[2] - </span><br>
<span id="reference-3">[3] - </span><br>
<span id="reference-4">[4] - </span><br>
<span id="reference-5">[5] - </span><br>
<span id="reference-6">[6] - </span><br>
<span id="reference-7">[7] - </span><br>
<span id="reference-8">[8] - </span><br>
<span id="reference-9">[9] - </span><br>