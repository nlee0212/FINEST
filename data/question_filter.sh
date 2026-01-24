export CUDA_VISIBLE_DEVICES=""

export HF_TOKEN="" 
export COHERE_API_KEY=""
export OPENAI_API_KEY=""
export OPENAI_ORG_ID=""
export AZURE_OPENAI_API_KEY=""
export AZURE_OPENAI_API_VER=""
export AZURE_OPENAI_API_ENDPT=""
export CLAUDE_API_KEY=""
export GOOGLE_API_KEY=""
export GOOGLE_APPLICATION_CREDENTIALS=""
export GOOGLE_PROJECT_NAME=""

# KOLD

python question_filter.py --dataset_dir "raw_data" \
                          --raw_dataset_file "kold_v1.json" \
                          --id_col "guid" \
                          --need_translation "False" \
                          --questionize_mode "kold" \
                          --model_name "gpt-4o" \
                          --model_path "gpt-4o-2024-05-13" \
                          --temperature 0\
                          --top_p 1 \
                          --is_pretrained "False" \
                          --gpt_azure "False"

# IBM-Rank-30k

python question_filter.py --dataset_dir "raw_data" \
                          --raw_dataset_file "arg_quality_rank_30k.csv" \
                          --need_translation "True" \
                          --questionize_mode "arg" \
                          --model_name "gpt-4o" \
                          --model_path "gpt-4o-2024-05-13" \
                          --temperature 0\
                          --top_p 1 \
                          --is_pretrained "False" \
                          --gpt_azure "False"

# SQuARe Train

python question_filter.py --dataset_dir "raw_data" \
                          --raw_dataset_file "square_response_train.json" \
                          --question_col "question" \
                          --need_translation "False" \
                          --model_name "gpt-4o" \
                          --model_path "gpt-4o-2024-05-13" \
                          --temperature 0\
                          --top_p 1 \
                          --is_pretrained "False" \
                          --gpt_azure "False"

# SQuARe Valid

python question_filter.py --dataset_dir "raw_data" \
                          --raw_dataset_file "square_question_valid.json" \
                          --question_col "question" \
                          --need_translation "False" \
                          --model_name "gpt-4o" \
                          --model_path "gpt-4o-2024-05-13" \
                          --temperature 0\
                          --top_p 1 \
                          --is_pretrained "False" \
                          --gpt_azure "False"