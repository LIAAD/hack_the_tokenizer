# LLM Concepts

- **Tokenization**: The process of breaking down text into smaller units (tokens) for processing by a language model.
- **Catastrophic forgetting**: The phenomenon where a model's performance on previously learned tasks deteriorates when it is retrained on new tasks.
- **Fertility**[\[5\]](#reference-5): Average number of tokens generated per word.
- **Byte Pair Encoding (BPE)**[\[6\]](#reference-6): A tokenization algorithm that iteratively merges the most frequent pairs of tokens in a vocabulary.
- **Continued Pretraining (CPT)**[\[7\]](#reference-7): A method of adapting a pre-trained language model to a new language by continuing to train it on a new dataset.
- **Encoder LLM**: An LLM that simply tries to find the correct word for a given context (E.G: `What is the [MASK] everyone loves?` -> The LLM would try to find the correct word for the [MASK] token).
- **Decoder LLM**: An LLM that tries to predict the next word(s) in a sequence (E.G: `What is the food everyone loves?` -> The LLM would try to predict the next word(s) in the sequence).