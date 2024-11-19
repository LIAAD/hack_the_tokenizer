# Introduction

This document summarizes the paper [Sabiá: Portuguese Large Language Models](https://export.arxiv.org/pdf/2304.07880v4.pdf).

# Sabiá: Portuguese Large Language Models

## Summary
This paper focuses on the development of a Portuguese LLM by further pretaining existing mostly english-focused trained models.

This training lowers the main models' performance in English tasks, which shows the compromise in refining models for other domains.


## Approach

The authors mainly focused on:
1. Picking Portuguese texts (datasets)
    1. Subset of the ClueWeb[\[1\]](#reference-1)[\[2\]](#reference-2)
1. Modifying said texts to accommodate the specific requirements of the Portuguese Language. Namely:
    - Quality Filter from MassiveText[\[3\]](#reference-3)
    - ftfy (normalization fixes *mojibakes* and remove remnant HTML tags)
    - converting wikitexts to human-readbale ones
    - excluding docs containing less than 200 unique tokens
1. Evaluating the models using Poeta (Portuguese Evaluation Tasks):
    - Native - Datasets in "POETA" originally written in PT-BR
    - Translated - Datasets in "POETA" originally written in English and then translated to PT-BR
1. Using Normalized Preferred Metric (NPM) as the primary evaluation Measure.
$$ NPM = \frac{1}{N}\sum_{i=1}^{N}100\times\frac{\text{[raw preferred metric]}_i - \text{[random score]}_i}{\text{[high score]}_i - \text{[random score]}_i} $$

## Main findings

All 3 new trained models (Sabiá-J, Sabiá-7B and Sabiá-65B) outperformed the original models in the Portuguese language. \ 
The larger model outperformed GPT3.5 in the Native and Translated datasets

## Evaluation Methodology

Most of the datasets in POETA where either Multi-choice questions. \ 
Some models were having difficulties choosing an answer and so they opted for \ 
an approach similar to XGLM authors (calculating likelihood of each choice, and picking the option with the highest probability)

One of the datasets, FaQuAD [\[4\]](#reference-4) - the only dataset without predetermined answers in the dataset, they allowed the models to generate answers.

## Results

All trainings showed a significant improvement in the Portuguese language. \
Although this improvement was made, it may exist some data conatamination (meaning that the model was trained on the same data as the test data) for some of the results (aside from ENEM 2022 which had not been relased at the time of the experiment).

# Conclusion

- Further Pretraining on a target language vastly increases the performance of the model in that language.
- The training can lower performance in the original language.
- The larger the model, the better the performance.

# References
<span id="reference-1">[1] - Overwijk, A., Xiong, C., Callan, J.: Clueweb22: 10 billion web documents with rich information. In: Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. pp. 3360–3362 (2022)</span><br>
<span id="reference-2">[2] - Overwijk, A., Xiong, C., Liu, X., VandenBerg, C., Callan, J.: Clueweb22: 10 billion web documents with visual and semantic information (2022)</span><br>
<span id="reference-3">[3] - Rae, J.W., Borgeaud, S., Cai, T., Millican, K., Hoffmann, J., Song, F., Aslanides, J., Henderson, S., Ring, R., Young, S., et al.: Scaling language models: Methods, analysis & insights from training gopher. arXiv preprint arXiv:2112.11446 (2021)</span><br>
<span id="reference-4">[4] - </span><br>
<span id="reference-5">[5] - </span><br>
<span id="reference-6">[6] - </span><br>
<span id="reference-7">[7] - </span><br>
<span id="reference-8">[8] - </span><br>
<span id="reference-9">[9] - </span><br>