# Introduction

This document summarizes the paper ["Exploring Design Choices for Building Language-Specific LLMs"](https://arxiv.org/pdf/2406.14670).

# Exploring Design Choices for Building Language-Specific LLMs

Github project is available [here](https://github.com/atutej/token-language-adaptation)

## Summary
Paper explores the adaptation of a pre-trained language model using new tokenizers and adapting the pre-trained language model to the new language using CPT (Continued Pretraining).


## Approach

This paper explores the adaptation of a pre-trained language model. 
It mainly focuses on three major design choices:
1. Base LLM 
1. Size of the augmented Vocabulary
1. Amount of CPT (Continued Pretraining Data)

They evaluate how each choice impacts the final task performances on four diverse languages (Hindi, Turkish, Arabic and Tamil).<br>
They tested on 7 different base LMs.

## Main findings
- Base LM performance is not a good indicator of the final task performance. (With the correct tokenizer/training, the performance of a base LM can be drastically improved)
- A small vocabulary addition leads to great efficiency gains.
- Extending vocabs may drop end task performance, but this can be recovered by making use of CPT.
- Initialization of nre token parameters is important for efficient adaptation.

## Methodology
### Augmenting Token Vocabulary

1. **Generating Target Language Tokens** - BPE using 300K examples on the target language (generates vocab $V'$ of size 1K to 50K)
1. **Merging with Original Vocabulary** - Let $\Delta V=V'-V$ which is the non-overlapping tokens of $V'$ in $V$. Then, we update $V$ with: $V_{new} = V \oplus \Delta V$ where $\oplus$ is the concatenation operation.
    1. This assumes the Frequency of the first `new` token is lower than the frequency of the `old` token in the BPE merging procedure.

### Integrating New Tokens to the LLM
1. **Embedding Initialization** - \<DIDN'T FULLY UNDERSTAND THIS STEP, ASK FOR A BIT OF CLARIFICATION AND TO CHECK IF IT IS CORRECT\> Token embedding $E(v)$ is obtained using, for each new token $v \in \Delta V$:
    1. Find original tokenization of $v$ in the original tokenizer $V$ ($t = Tokenize(v, V)$)
    1. Now we know $t = \{t_0, t_1, ..., t_n\}$ for some $n\in \mathbb{N}$  where $t_i \in V, \forall i \in \{0, 1, ..., n\}$
    1. The embedding $E(v)$ is obtained by averaging the embeddings of the tokens in $t$: $E(v) = \frac{1}{n} \sum_{i=1}^n E(t_i)$
1. **Continued Pretraining** - The CPT was trained with 200K examples (~200M tokens) and 500K examples (`~500M tokens`) for larger 
1. **Implementation Details** - For 200K examples, it took them 18h of training on 4 A40 GPUs (reference point). They utilized full fine-tunning in all of their experiments, since training with LoRA yielded worse performance and only led to 1.5x less compute. \<NEED CLARIFICATION, WHAT IS LoRA and what is "utilized full fine-tuning" in this context?\>

## Results
- Found Significant disparity in fertility between English and other languages, before vocabulary adaptation, even in some commercial models with multilingual capabilities.
- Augmenting the vocab with ~10K language-specific tokens, significantly mitigates this disparity.
- The correlation between "added tokens" and "fertility" starts to diminish as the number of added tokens increases. (So it brings diminishing returns). Adding ~1K tokens doubles the fertility on average.
- Adapted models to other languages, can match performance of other Multi-Lingual models.
- Adapting monolingual model trained on a highly curated data might be more challenging as they couldn't get good results from adapting those.
- Smaller multilingual models show improved performance, but the larger multilingual models don't show notable performance gains.
- LLM may lose performance in original language after adaptation (even if just partly).



# Conclusion
- Explores different Embedding Initialization techniques: [MEAN, RANDOM-INIT, RANDOM-Tok-Emb, FOCUS, Learned-Emb].
    - Might be interesting to explore these techniques to visualize which brings better and faster results.
- Supports similar work on tokenizer adaptation, which will be interesting to explore.
- Larger models might not be as "Adaptive" as smaller models.
    - May be interesting to explore the impact the size of the model has on adaptation. How can we determine if a model is worth it to be adapted?


# Takeaways

# References
<span id="reference-1">[1] - \<SIMILAR WORK - THEY DID SIMILAR WORK TO THIS PAPER\> Gautier Dagan, Gabriel Synnaeve, and Baptiste Rozière. 2024. [Getting the most out of your tokenizer for pre-training and domain adaptation](https://arxiv.org/abs/2402.01035). Preprint, arXiv:2402.01035.</span><br>
<span id="reference-2">[2]</span><br>
<span id="reference-3">[3]</span><br>
<span id="reference-4">[4]</span><br>
<span id="reference-5">[5]</span><br>
<span id="reference-6">[6]</span><br>
<span id="reference-7">[7]</span><br>
<span id="reference-8">[8]</span><br>
<span id="reference-9">[9]</span><br>