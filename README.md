# Hack the Tokenizer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python library for augmenting pretrained language model tokenizers to handle non-English languages more effectively, with a focus on Portuguese.

## Motivation

Language models have revolutionized AI applications across domains, but they face significant challenges with non-English languages. This disparity stems from tokenization inefficiency:

- **Quality Gap**: Non-English text generation often results in lower quality output
- **Cost Disparity**: Processing non-English text is more expensive due to higher token counts

For example, the Llama-3 tokenizer breaks down the English sentence "This is a thesis proposal!" into 6 tokens, while the equivalent Portuguese sentence "Isto é uma proposta de tese!" produces 10 tokens. This inefficiency:

1. Reduces generation quality (more tokens = more opportunities for errors)
2. Increases API costs (pricing is typically per token)

## Approach

This project develops a method to augment pretrained tokenizers with language-specific tokens, improving efficiency for non-English languages without requiring full model retraining:

1. **Token Generation**: Create new language-specific tokens
2. **Embedding Initialization**: Initialize embeddings for new tokens using various strategies
3. **Fine-tuning**: Optimize the new token embeddings
4. **Evaluation**: Measure improvements using benchmarks and fertility metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hack-tokenizer.git
cd hack-tokenizer

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Requirements

- Python 3.10+
- PyTorch
- Transformers
- Other dependencies listed in requirements.txt

## Usage

### Quick Start

```python
from hack_tokenizer.hack import ModelHacker
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a ModelHacker instance
hacker = ModelHacker(
    dataset=["your Portuguese text corpus here"],
    batch_size=8,
    learning_rate=1e-6
)

# Hack the tokenizer with 1000 new Portuguese tokens
model, tokenizer = hacker.hack(
    model=model,
    tokenizer=tokenizer,
    encoding_tokenizer=tokenizer,
    num_tokens=1000,
    embed_initializer_method="mean",
    show_progress=True,
    train=True
)

# Save the hacked model and tokenizer
model.save_pretrained("./hacked-model")
tokenizer.save_pretrained("./hacked-tokenizer")
```

### CLI Usage

```bash
# Run tokenizer hacking with default parameters
python -m hack_tokenizer.hack --model microsoft/phi-2 --num-tokens 1000

# Run evaluation on hacked model
python -m hack_tokenizer.evaluation --model ./hacked-model --tokenizer ./hacked-tokenizer
```

## Project Structure

```
hack_tokenizer/
├── benchmark/            # Benchmark implementations
│   ├── base.py           # Base benchmark class
│   ├── CalamePT.py       # Portuguese benchmark
│   ├── MMLU.py           # Multilingual benchmark
│   └── SuperGLUE.py      # English benchmark
├── evaluation/           # Evaluation framework
│   └── evaluation.py     # Main evaluation logic
├── hack/                 # Core tokenizer hacking functionality
│   ├── ModelHacker.py    # Model embedding manipulation
│   └── TokenizerHack.py  # Tokenizer modification
├── metrics/              # Evaluation metrics
│   ├── base.py           # Base metric class
│   ├── FertilityBoost.py # Fertility improvement metric
│   ├── FertilityInput.py # Input tokenization efficiency
│   ├── FertilityOutput.py # Output tokenization efficiency
│   └── Perplexity.py     # Language modeling quality
└── utils/                # Utility functions
    ├── cli.py            # Command-line interface
    ├── DatasetClass.py   # Dataset handling
    ├── functions.py      # Helper functions
    └── loader.py         # Model loading utilities
```

## Results

Our approach demonstrates significant improvements for Portuguese language processing:

- **Token Efficiency**: Reduced token count by 15-30% for Portuguese text
- **Generation Quality**: Improved coherence and fluency in Portuguese text generation
- **Cost Reduction**: Lower token counts translate to reduced API costs
- **Model Performance**: Maintained or improved performance on Portuguese benchmarks

The most effective embedding initialization strategy was mean initialization, which outperformed weighted average and translation-based approaches.

## Benchmarks

The project includes several benchmarks to evaluate performance:

- **CalamePT**: Portuguese language benchmark
- **MMLU**: Multilingual benchmark with Portuguese subset
- **SuperGLUE**: English benchmark to verify no regression in original language

To run benchmarks:

```bash
python -m hack_tokenizer.evaluation --benchmark calamept --model ./hacked-model
```

## Contributing

Contributions are welcome! Here are some areas for future work:

1. Support for additional languages beyond Portuguese
2. Alternative embedding initialization strategies
3. Integration with modern LLM deployment frameworks (vLLM)
4. Performance optimization for larger models

Please follow these steps to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{pinto2025hack,
  author = {Pinto, Duarte},
  title = {Hack the Tokenizer: Augmenting Pretrained Language Models for Non-English Languages},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/hack-tokenizer}}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This research builds upon work from "Efficiently Adapting Pretrained Language Models to New Languages" (https://arxiv.org/pdf/2311.05741)
- Thanks to Microsoft for the Phi model series used in this research
