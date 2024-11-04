# Introduction

This document summarizes the paper ["Efficient and Effective Text Encoding for Chinese Llama and Alpaca"](https://arxiv.org/pdf/2406.14670).

# Efficient and Effective Text Encoding for Chinese Llama and Alpaca

Github project is available in:
- [Llama-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
- [LLaMa-Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)

## Summary
Main goal of the paper is to improve the performance of the Llama and Alpaca models for Chinese language tasks.<br>
This is done using open-source alternatives so as to open up the field for the research community.<br>
They propose the development of Chinese LLaMA and Alpaca models with enhanced capabilities for understanding and generating Chinese content.<br>
Add new 20K Chinese tokens to the vocabularies of LLama and Alpaca Models.<br>
Fine-tuning using LoRA approach.

## Approach
- Training chinese Tokenizer with SentencePiece on Chinese Corpora with a vocabulary size of 20K.
- Merge the new tokens into the original tokenizer (by taking the Union of the two vocabularies).
- Resizing of the word embeddings to addapt the LLaMA for the Chinese LLaMA model. Resizes from shape $V \times H$ to $V' \times H$ where $V = 32,000$ denotes the original vocabulary size, and $V' = 49,953$ is the new vocabulary size of the Chinese LLaMA tokenizer. 
    - Extra rows are simply appended to the end of the original embedding matrices, so that the original embeddings remain unaffected.
- Fine-tuning using LoRA approach.
    - Due to its efficiency in low-resource environments, LoRA is used for fine-tuning.
    - LoRA freezes the existing pre-trained model weights and injects treinable low-rank matrices into each layer. This approach significantly reduces total trainable parameters.

## Results
- Tokens generated shrink in ~2 times. (from 35 down to 16, for example)
- model performance improved in all aspects for the chinese language

# Conclusion
This is more of a migration paper than fine-tuning one.<br>
Main goal was to create a "Chinese" version of the Llama and Alpaca models, which is a bit different than what we aim to achieve.<br>
Some of it's findings and details may still be interesting  in our training procedures.


# References
<span id="reference-1">[1] - EFFICIENT AND EFFECTIVE TEXT ENCODING FOR CHINESE LLAMA AND ALPACA</span><br>