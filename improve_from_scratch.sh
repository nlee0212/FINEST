#!/bin/bash

export CUDA_VISIBLE_DEVICES=""

export HF_TOKEN="" 
export COHERE_API_KEY=""
export OPENAI_API_KEY="OPENAI_API_KEY"
export OPENAI_ORG_ID="OPENAI_API_KEY"
export AZURE_OPENAI_API_KEY=""
export AZURE_OPENAI_API_VER=""
export AZURE_OPENAI_API_ENDPT=""
export CLAUDE_API_KEY=""
export GOOGLE_API_KEY=""
export GOOGLE_APPLICATION_CREDENTIALS=""
export GOOGLE_PROJECT_NAME=""

file_names=$(find data/responses -name "*numbered_response.csv" -exec basename {} \;)
for file in $file_names; do
    echo $file
done

for file in $file_names; do
    python improve.py --response_dir "data/responses/" \
                       --response_file ${file} \
                       --improve_dir "data/improved/" \
                       --improve_file "${file%_numbered_response.*}_improved.csv" \
                       --question_col "question" \
                       --id_col "guid" \
                       --response_col "numbered_response" \
                       --need_eval "True" \
                        --mode "score" \
                        --include_short_def "True" \
                        --include_def "True" \
                        --model_name "GPT-4o" \
                        --model_path "gpt-4o-2024-05-13" \
                        --temperature 1 \
                        --top_p 0.9 \
                        --is_pretrained "False" \
                        --gpt_azure "False"
    
    python improve.py --response_dir "data/responses/" \
                       --response_file ${file} \
                       --improve_dir "data/improved/" \
                       --improve_file "${file%_numbered_response.*}_improved.csv" \
                       --question_col "question" \
                       --id_col "guid" \
                       --response_col "numbered_response" \
                       --need_eval "True" \
                        --mode "error" \
                        --include_short_def "True" \
                        --include_def "True" \
                        --model_name "GPT-4o" \
                        --model_path "gpt-4o-2024-05-13" \
                        --temperature 1 \
                        --top_p 0.9 \
                        --is_pretrained "False" \
                        --gpt_azure "False"

done