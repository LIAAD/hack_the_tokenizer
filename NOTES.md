# Evaluation helpers
- Benchmark to classify multiple languages automatically - https://github.com/EleutherAI/lm-evaluation-harness
- Try to do a evaluation using https://github.com/vllm-project/vllm to speed-up evaluation process
- Use GlobalMMLU to try and evaluate our model with PT (https://arxiv.org/pdf/2412.03304)

# Papers to explore
- [TokenFormer: Rethinking Transformer Scaling with Tokenized Model Parameters](https://arxiv.org/abs/2410.23168)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- **Gobal MMLU**
    - Paper: https://arxiv.org/pdf/2412.03304
    - Twitter post: https://x.com/singhshiviii/status/1864695486999110054?mx=2
- **Motivational papers**
    - [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/pdf/2305.15425)
    - [XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models](https://aclanthology.org/2023.emnlp-main.813/)
    - [MYTE: Morphology-Driven Byte Encoding for Better and Fairer Multilingual Language Modeling](https://aclanthology.org/2024.acl-long.804/)


# Papers/stuff to keep as reference
- ["Efficiently Adapting Pretrained Language Models to New Languages"](https://arxiv.org/pdf/2311.05741)
    - Pretty similar work to what we're trying to achieve, they only focused on the "mean"
- ["Getting the most out of your tokenizer for pre-training and domain adaptation"](https://arxiv.org/pdf/2402.01035)
    - Good paper to understand which procedures to follow on the pre-tokenization step (compression with reg. exp/BPE/etc)
- **Tokenizer training**
    - https://github.com/huggingface/tokenizers




# Future work
- Calculation of new Embedding:
    1. Use weigthed average based on the initial tokens lengths
        - E.g: New token: "martelo". Old tokens: ["mar", "telo"] -> the embed for "martelo" will be the weighted average of the embeds for "mar" and "telo", with 3/7 weighted for "mar" and 4/7 weighted for "telo"
    1. Try to translate the meaning of the word to the Original Language from the Target Language, and use the vector for that translated word
        - E.g: New token: "martelo". Translation is "hammer", so use the embed for the word "hammer" (if the original language is English)
- Final deployment:
    1. Explore https://github.com/vllm-project/vllm to deploy a final LLM or to do some testings

