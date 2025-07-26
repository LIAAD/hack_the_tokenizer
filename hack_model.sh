#!/bin/bash

# Configuration variables
export model="Qwen/Qwen2.5-1.5B-Instruct"
export device=cuda
export batch=8
export temperature=0.8
export learning_rate=1e-6
export number_new_tokens=1_000
export dataset_tokenizer="./data/tokenizer_pt-pt.txt"
export dataset_training=""
# export dataset_training="./data/calamept_dataset.txt"
export embed_init_method=weighted_drop(1.5)
export max_new_tokens=50
export stop_words="<end>,</end>,<|endoftext|>"
export hack_model=false

python -m hack_tokenizer.hack \
	--model $model \
	--device $device \
	--batch $batch \
	--temperature $temperature \
	--learning_rate $learning_rate \
	--number_new_tokens $number_new_tokens \
	--dataset_tokenizer $dataset_tokenizer \
	--dataset_training $dataset_training \
	--embed_init_method $embed_init_method \
	--max_new_tokens $max_new_tokens \
	--stop_words $stop_words \
	--hack_model $hack_model \