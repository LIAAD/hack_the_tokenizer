# Hack the Tokenizer

Language-model-based tools have gained popularity since the introduction of ChatGPT. This is no surprise given the profound impact intelligent machines can have on society. Whether in the education, medical, or legal domain, models that can enhance or even replace experts could significantly lower the cost of services and, consequently, democratize access to them. However, it is crucial to avoid exacerbating existing inequalities and ensure these tools benefit everyone.  

This potential issue arises because training and developing language models require substantial financial resources, predominantly available to large tech companies focused on generating revenue. Consequently, a disparity can already be observed: since English is the most widely spoken language, companies prioritize developing English-based language models, offering limited support for other languages. This results in a quality gap, where generating non-English text with such models often results in less quality with higher costs. Both issues that can be linked to the tokenizer of the language model. To understand how a (brief) tokenizer explanation is in order.  

At a high level, a tokenizer is an object that encodes the text into a sequence of numbers that the language model can process. State-of-the-art tokenizers (such as Byte Pair Encoding or SentencePiece) are optimized in the training corpus so that they break the text as effectively as possible. In practice, this means breaking the text as little as possible while maintaining a good coverage. Since English is the language that is more represented in the training corpus, the tokenizer ends up including more tokens in the vocabulary that were derived from English text. Consequently, non-English texts ends up being split into more tokens than English texts with the same content. As an example, the Llama-3 tokenizer breaks down the sentence “This is a thesis proposal!” into six tokens, but the same sentence in Portuguese “Isto é uma proposta de tese!” ends up producing ten tokens. 

Naturally, this directly affects the generative effectiveness of the language model, which is trained to predict the next token given a sequence of tokens. If the sentence is broken down into more tokens, it gives more room for errors which will then cascade to produce even more errors. Furthermore, since the cost of using a language model is associated with the number of tokens processed, non-English text becomes more expensive to process. 

The goal of this thesis is to develop an approach to augment the tokenizer of a pretrained language model to handle non-English texts more effectively. This research will use small language models (from the Phi family) and aim to improve the quality of text produced in European Portuguese by adding specific tokens to the tokenizer. Embeddings for these new tokens will have to be added to the embeddings table of the model. This should be the main challenge of the research. How can we do that?  

## References 
- This idea has been explored (to some extent) for the “Hungarian and Thai” languages [Efficiently Adapting Pretrained Language Models to New Languages](https://arxiv.org/pdf/2311.05741)
- Some preliminary analysis that supports this thesis is available in this [notebook](https://github.com/hmosousa/tokens/blob/main/notebooks/main.ipynb) 
- A [good post](https://leimao.github.io/blog/Byte-Pair-Encoding/) explaining Byte Pair Encoding with code 
- The Byte Pair Encoding paper: [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909)  
- The SentencePiece paper: [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/pdf/1808.06226)  
- Phi models from microsoft: [Phi-1](https://huggingface.co/microsoft/phi-1), [Phi-1.5](https://huggingface.co/microsoft/phi-1_5), [Phi-2](https://huggingface.co/microsoft/phi-2), and [Phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) 

## Objectives 
- Develop an approach to add new tokens to a pretrained tokenizer 
- Evaluate the proposed approach  
- Write and submit a paper to a conference 

## Contributions 
- A method that effectively adds tokens to pretrained tokenizers  

## Timeline
- Janeiro: Have the related Work
- Abril: End of research. Start writing the thesis
- June: Thesis defence
