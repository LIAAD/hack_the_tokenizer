# Introduction

This document summarizes the paper [Advancing Generative AI for Portuguese with Open Decoder Gervásio PT*](https://arxiv.org/abs/2402.18766).


# Advancing Generative AI for Portuguese with Open Decoder Gervásio PT*

## Summary

The paper presents a new portuguese decoder (Gervásio PT*), a portuguese LLM. \
This model was pre-trained from LLaMA 2 7B model as a starting point. \
By using the LLaMA 2 7B Model, they created the first 7B decodear LLM for PT-PT (and they also created a PT-BR variant).

In their related work section, they provide with information as to WHY they chose LLaMA 2 7B versurs other competitive models. Can be a good reference/baseline for this dissertation.

## Approach

The authors mainly focused on:
1. Obtaining the datasets + Selecting tasks
    1. Task Selection: They tried to select those tasks that, when translated to PT, preserved the linguistic properties of the task.
    1. Task Translation: They utilized DeepL, a transalation tool.
    1. Task Templates: Instructions were manually crafted for each task.
    1. Training Data: Machine translated data using DeepL from the datasets: STS-B, WNLI (From GLUE) and BoolQ, CB and MultiRC (from SuperGLUE).`
    1. Testing Data: Some translated datasets were saved for testing - MRPC, RTE (from GLUE) and COPA  (from SuperGLUE).
1. Fine-tuning LLaMA 2 7B with Zero-out technique - While the entire prompt received attention during fine-tuning, only the response tokens were subjected to back-propagation.
## Main findings

Greatest performance than the LLaMA 2 7B base model. \
The PT-BR variant out-performed the PT-PT version, possibly due to the fact that the translations of the training dataset and test dataset having a slight bias towards PT-BR. \
Unfortunately there still is a lack of PT-PT datasets.


## Evaluation Methodology

Translated datasets from English to PT-PT and PT-BR:
- MRPC (Similarity) - From GLUE
- RTE (inference) - From GLUE
- COPA (reasoning/qa) - From SuperGLUE


# Conclusion
They created a model based on translated pieces... May be a good model to compare ours to, but it doesn't seem to have a good starting-point for us to pick up some data from.

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