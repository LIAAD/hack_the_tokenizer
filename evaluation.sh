python -m hack_tokenizer.evaluation \
    --model HuggingFaceTB/SmolLM2-135M\
    --device cuda \
    --batch 8 \
    --learning_rate 1e-6 \
    --number_new_tokens 1000 \
    --output_directory ./outputs \
    --embed_init_method "weighted_drop(1.5)"
    # --model Qwen/Qwen2.5-1.5B-Instruct\
    # --temperature None \
    # --dataset_tokenizer None \
    # --dataset_training None \