python -m hack_tokenizer.evaluation \
    --model HuggingFaceTB/SmolLM2-135M \
    --device cuda \
    --batch 8 \
    --learning_rate 1e-6 \
    --number_new_tokens 1000 \
    --dataset_tokenizer './data/tokenizer_pt-pt.txt' \
    --dataset_training './data/calamept_dataset.txt' \
    --output_directory ./outputs \
    --embed_init_method "weighted_drop(1.5)" \
    --datasets_metrics 'Fertility=./data/metrics_evaluation_dataset.txt,Perplexity=./data/metrics_evaluation_dataset.txt'

# python -m hack_tokenizer.evaluation \
#     --model Qwen/Qwen2.5-1.5B-Instruct \
#     --device cuda \
#     --batch 8 \
#     --learning_rate 1e-6 \
#     --number_new_tokens 1000 \
#     --dataset_tokenizer './data/tokenizer_pt-pt.txt' \
#     --dataset_training './data/calamept_dataset.txt' \
#     --output_directory ./outputs \
#     --embed_init_method "weighted_drop(1.5)" \
#     --datasets_metrics 'Fertility=./data/metrics_evaluation_dataset.txt,Perplexity=./data/metrics_evaluation_dataset.txt'
