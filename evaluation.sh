#!/bin/bash

# Configuration variables
export number_new_tokens=10000
export device=cuda
export batch=8
export learning_rate=1e-6
export number_new_tokens=$number_new_tokens
export dataset_tokenizer='./data/tokenizer_pt-pt.txt'
export dataset_training='./data/calamept_dataset.txt'
export output_directory=./outputs
export embed_init_method="weighted_drop(1.5)"
export datasets_metrics='Fertility=./data/metrics_evaluation_dataset.txt,Perplexity=./data/metrics_evaluation_dataset.txt'

# List of models to evaluate
MODELS=(
    "HuggingFaceTB/SmolLM2-135M"    # Small     SingleModal     Model
    "Qwen/Qwen2.5-1.5B-Instruct"    # Medium    MultiModal      Model
)

# Loop through each model and run evaluation
for model in "${MODELS[@]}"; do
    echo "Evaluating model: $model"
    
    python -m hack_tokenizer.evaluation \
        --model "$model" \
        --device $device \
        --batch $batch \
        --learning_rate $learning_rate \
        --number_new_tokens $number_new_tokens \
        --dataset_tokenizer $dataset_tokenizer \
        --dataset_training $dataset_training \
        --output_directory $output_directory \
        --embed_init_method $embed_init_method \
        --datasets_metrics $datasets_metrics
done