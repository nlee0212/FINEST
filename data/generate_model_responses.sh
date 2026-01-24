#!/bin/bash

export CUDA_VISIBLE_DEVICES="GPU_NUMBERS"

export HF_TOKEN="" 
export COHERE_API_KEY=""
export OPENAI_API_KEY="OPENAI_API_KEY"
export OPENAI_ORG_ID="OPENAI_ORG_ID"
export AZURE_OPENAI_API_KEY=""
export AZURE_OPENAI_API_VER=""
export AZURE_OPENAI_API_ENDPT=""
export CLAUDE_API_KEY=""
export GOOGLE_API_KEY="GOOGLE_API_KEY"
export GOOGLE_APPLICATION_CREDENTIALS=""
export GOOGLE_PROJECT_NAME=""

declare -A MODEL_PATHS
MODEL_PATHS["GPT-4o"]="gpt-4o-2024-05-13"
MODEL_PATHS["Gemini-1_5-Pro"]="gemini-1.5-pro-latest"
MODEL_PATHS["Orion-14B-Chat"]="OrionStarAI/Orion-14B-Chat"

declare -A IS_PRETRAINED
IS_PRETRAINED["GPT-4o"]="False"
IS_PRETRAINED["Gemini-1_5-Pro"]="False"
IS_PRETRAINED["Orion-14B-Chat"]="True"

QUESTION_FILES=("arg_quality_rank_30k_final.csv" "kold_v1_final.csv" "square_response_train_final.csv" "square_question_valid_final.csv")

for question_file in "${QUESTION_FILES[@]}"; do
    for model_name in "${!MODEL_PATHS[@]}"; do
        model_path=${MODEL_PATHS[$model_name]}
        is_pretrained=${IS_PRETRAINED[$model_name]}

        python generate_responses.py --question_dir "final_questions" \
                                     --question_file ${question_file} \
                                     --response_dir "responses" \
                                     --response_file "${question_file%.*}_${model_name}_response.csv" \
                                     --id_col "guid" \
                                     --question_col "question" \
                                     --dataset ${question_file%.*} \
                                     --model_name ${model_name} \
                                     --model_path ${model_path} \
                                     --model_cache_dir "../.cache" \
                                     --temperature 0 \
                                     --top_p 1 \
                                     --is_pretrained ${is_pretrained} \
                                     --gpt_azure "False"

        python post_process_responses.py --response_dir "responses" \
                                        --response_file "${question_file%.*}_${model_name}_response.csv" \
                                        --numbered_response_file "${question_file%.*}_${model_name}_numbered_response.csv" \
                                        --id_col "guid" \
                                        --question_col "question" \
                                        --model_name "GPT-4o" \
                                        --model_path "gpt-4o-2024-05-13" \
                                        --temperature 0 \
                                        --top_p 1 \
                                        --is_pretrained "False" \
                                        --gpt_azure "False"
    done
done

