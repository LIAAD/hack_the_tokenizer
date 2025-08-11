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
export output_format='parquet'
export embed_init_method="weighted_drop(1.5)"
export datasets_metrics='FertilityOutput=./data/fertility_output_evaluation-dataset.txt,FertilityInput=./data/metrics_evaluation_dataset.txt,Perplexity=./data/metrics_evaluation_dataset.txt'

# List of models to evaluate
MODELS=(
    "HuggingFaceTB/SmolLM2-135M"    # Small     SingleModal     Model
	"Qwen/Qwen2.5-1.5B-Instruct"    # Medium    MultiModal      Model
    "HuggingFaceTB/SmolLM3-3B"		# Big		MultiModal		Model
)

# Number of tokens to iterate over
NUM_TOKENS=(
    1000
    5000
    7500
)

# Loop through each model and run evaluation
for model in "${MODELS[@]}"; do
    echo "Evaluating model: $model"
    for number_new_tokens in "${NUM_TOKENS[@]}"; do 
        echo "Number New Tokens: $number_new_tokens"	    
	python -m hack_tokenizer.evaluation \
 	   --model "$model" \
	   --device $device \
	   --batch $batch \
	   --learning_rate $learning_rate \
	   --number_new_tokens $number_new_tokens \
	   --dataset_tokenizer $dataset_tokenizer \
	   --dataset_training $dataset_training \
	   --output_directory $output_directory \
	   --output_format $output_format \
	   --embed_init_method $embed_init_method \
	   --datasets_metrics $datasets_metrics
    done
done
