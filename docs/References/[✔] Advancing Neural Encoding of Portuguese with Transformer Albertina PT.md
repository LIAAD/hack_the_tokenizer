# Introduction

This document summarizes the paper [Advancing Neural Eencoding of Portuguese with Transformer Albertina PT-*](https://arxiv.org/pdf/2305.06721).

# Advancing Neural Eencoding of Portuguese with Transformer Albertina PT-*

## Summary

Paper builds a portuguese (PT-PT and PT-BR) large language madel, using DeBERTa[\[1\]](#reference-1) as a starting point.

This model was trained on datasets gathered for PT-PT and PT-BR in addition to the brWaC corpus[\[2\]](#reference-2) for PT-BR.

## Approach

Training of 2 models: Albertina PT-PT and Albertina PT-BR.
The following steps were followed:
1. Use DeBERTa as a starting point model.
1. Before pre-training, data was filtered using the BLOOM pre-processing pipeline[\[7\]](#reference-7)
1. Apply a Pre-training step to the models:
    1. PT-BR: Pre-training with the brWaC corpus (2.7 billion tokens, 3.5 million documents)
    1. PT-PT: Pre-training using some openly available corpora of European Portuguese (2.2 billion tokens, 8 million documents) from:
        - OSCAR[\[3\]](#reference-3)
        - DCEP[\[4\]](#reference-4)
        - Europarl[\[5\]](#reference-5)
        - ParlamentoPT[\[6\]](#reference-6)
1. After pre-training step, fine-tuning for downstream tasks was performed:
    1. Datasets organized into 2 groups: G1 ASSIN 2 benchmark datasets[\[8\]](#reference-8) / G2 Translated datasets from english used in the GLUE benchmark [\[9\]](#reference-9)
    1. ASSIN 2 was not translated from PT-BR to PT-PT due to the possibility of degradation to the quality of the dataset (read paper section 3.2 ASSIN 2 part)
    1. PT-PT model was only fine-tuned with the GLUE translated dataset


## Main findings

This paper has some good references and it can also be refered to in the "Related Work" section. \ 
They don't provide with detailed information related to our work, but they do provide interesting insights on datasets used and methodologies used.

## Evaluation Methodology
For the PT-PT model, only the GLUE translated dataset was used for evaluation. \
The PT-BR variant, used ASSIN 2 and GLUE (ASSIN 2 is originally in PT-BR and they did not translate it to PT-PT. GLUE was also translated from english to PT-BR).


# References
<span id="reference-1">[1] - DeBERTa: Decoding-enhanced BERT with Disentangled Attention ([git](https://github.com/microsoft/DeBERTa)) </span><br>
<span id="reference-2">[2] - BrWaC (Brazilian Portuguese Web as Corpus) ([huggingface](https://huggingface.co/datasets/UFRGS/brwac))</span><br>
<span id="reference-3">[3] - OSCAR [Abadji et al., 2022] Abadji, J., Ortiz Suarez, P., Romary, L., and Sagot, B. (2022). Towards a cleaner document-oriented multilingual crawled corpus. In Proceedings of the Thirteenth Language Resources and Evaluation Conference (LREC), pages 4344–4355</span><br>
<span id="reference-4">[4] - DCEP [Hajlaoui et al., 2014] Hajlaoui, N., Kolovratnik, D., Väyrynen, J., Steinberger, R., and Varga, D. (2014). DCEP-digital corpus of the European parliament. In Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC)</span><br>
<span id="reference-5">[5] - Europarl [Koehn, 2005] Koehn, P. (2005). Europarl: A parallel corpus for statistical machine translation. In Proceedings of Machine Translation Summit X: Papers, pages 79–86.</span><br>
<span id="reference-6">[6] - ParlamentoPT was collected from the Portuguese Parliament portal in accordance with its open data policy (https://www.parlamento.pt/Cidadania/Paginas/DadosAbertos.aspx and can be obtained here: https://huggingface.co/datasets/PORTULAN/parlamento-pt.)</span><br>
<span id="reference-7">[7] - BLOOM [Laurençon et al., 2022] Laurençon, H., Saulnier, L., Wang, T., Akiki, C., del Moral, A. V., Scao, T. L., Werra, L. V., Mou, C., Ponferrada, E. G., Nguyen, H., et al. (2022). The BigScience ROOTS corpus: A 1.6TB composite multilingual dataset. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track.</span><br>
<span id="reference-8">[8] - ASSIN 2 [Real et al., 2020] Real, L., Fonseca, E., and Gonçalo Oliveira, H. (2020). The ASSIN 2 shared task: a quick overview. In 14th International Conference on the Computational Processing of the Portuguese Language (PROPOR), pages 406–412. Springer</span><br>
<span id="reference-9">[9] - GLUE - This benchmark is freely distributed here: https://huggingface.co/datasets/PORTULAN/glue-ptpt </span><br>