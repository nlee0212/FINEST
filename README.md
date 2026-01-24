## FINEST: Improving LLM Responses to Sensitive Topics Through Fine-Grained Evaluation

This repository includes codes for dataset creation and experiments conducted for evaluating LLM responses across three aspects: content, appropriateness, and logic. 

## Dataset Construction

The FINEST dataset can be downloaded from [HuggingFace](https://huggingface.co/datasets/nayeon212/FINEST).

### Question Filtering
In this research, we use SQuARe, KOLD, and IBM-Rank-30k datasets for collecting sensitive questions. Especially for KOLD and IBM-Rank-30k, we questionize the comments or arguments for sensitive question collection. Then, we perform question filtering to leave out the high-quality and appropriate questions for our research.

```ruby
$ cd data
```

As we use LMs for questionizing and filtering the questions, API keys or other necessary information will be needed to run the models. Make sure to fill in your necessary information from below. You don't have to fill in all, but would have to fill in those that you only need.

```ruby
$ vi question_filter.sh

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
```

Then you will be able to run the following code to questionize & filter questions from the three datasets.
```ruby
$ sh question_filter.sh
```

If you want to apply the code to your own dataset, you may run the code below by filling in necessary fields.
```ruby
python question_filter.py --dataset_dir "DATASET_DIRECTORY" \
                          --raw_dataset_file "YOUR_DATASET (CSV, JSON, JSONL)" \
                          --id_col "COLUMN NAME WITH EACH DATA SAMPLE'S ID IF EXISTS (OPTIONAL)" \
                          --need_translation "IF THE DATASET NEEDS TRANSLATION (EN->KO)" \
                          --questionize_mode "CHOOSE BETWEEN ("kold" or "arg") IF QUESTIONIZATION NEEDED" \
                          --model_name "MODEL'S NAME" \
                          --model_path "MODEL'S PATH USED FOR API CALLING & DOWNLOADING FROM HUGGINGFACE" \
                          --temperature TEMPERATURE \
                          --top_p TOP_P \
                          --is_pretrained "IF THE MODEL IS A PRETRAINED MODEL THAT NEEDS TO BE DOWNLOADED FROM HUGGINGFACE" \
                          --gpt_azure "IF YOU USE AZURE API FOR GPT MODELS"
```
For more information regarding each of the arguments, please refer to `question_filter.py`. 

If you would like to construct your own prompt for questionizing, refer to `make_prompt_for_qg_kold()` or `make_prompt_for_qg_arg()` and modify `questionize()` in `question_filter_utlis.py`.

This process will end up with files including the final filtered questions within the `data/final_questions` directory.

### Model Response Generation
Then generate responses to the filtered questions generated from the previous stage. 
```ruby
python generate_responses.py --question_dir "final_questions" \
                             --question_file "question_file_name" \
                             --response_dir "responses" \
                             --response_file "response_file_name" \
                             --id_col "guid" \
                             --question_col "question" \
                             --dataset "dataset_name" \
                             --model_name "model_name" \
                             --model_path "model_path" \
                             --model_cache_dir "../.cache" \
                             --temperature 0 \
                             --top_p 1 \
                             --is_pretrained "True/False" \
                             --gpt_azure "False"

python post_process_responses.py --response_dir "responses" \
                                --response_file "response_file_name" \
                                --numbered_response_file "$numbered_response_file_name" \
                                --id_col "guid" \
                                --question_col "question" \
                                --model_name "model_name" \
                                --model_path "model_path" \
                                --temperature 0 \
                                --top_p 1 \
                                --is_pretrained "True/False" \
                                --gpt_azure "False"
```
By running the two codes, the model with `model_name` will generate the responses (Yes/No/General) to the questions from `question_dir/question_file` and save the responses to `response_dir/response_file`. After this process, the responses will be post-processed, including numbering and getting rid of too short responses.

You can simply run or refer to the following code to replicate our study.
```ruby
$ bash generate_model_responses.sh
```

## Evaluation
From this part, we run the codes in the main directory.

Based on our fine-grained taxonomy for response evaluation, the code below generates error-based evaluation feedback and score-based evaluation feedback of the given response.
```ruby
python evaluate.py --response_dir "data/responses/" \
                   --response_file "response_file" \
                   --evaluation_dir "data/evaluations/" \
                   --evaluation_file "evaluation_file" \
                   --question_col "question" \
                   --id_col "guid" \
                   --response_col "numbered_response" \
                   --model_name "GPT-4o" \
                   --model_path "gpt-4o-2024-05-13" \
                   --temperature 1 \
                   --top_p 0.9 \
                   --is_pretrained "False" \
                   --gpt_azure "False"
```
Row-by-row, the responses within the `response_dir/response_file`'s `response_col` column will be evaluated on our taxonomy, and the results will be saved at `evaluation_dir/evaluation_file`. 

You can simply run or refer to the code below for replication.
```ruby
$ bash evaluate.sh
```
You can also run evaluation on a single question & response pair by using the `eval_single_response()` function within `evaluate.py`.

The error-based and score-based evaluation results will be saved at `{response_col}_full_errors` and `{response_col}_full_scores` respectively, with some statistical evaluation results at `{response_col}_eval_summary` column.
The evaluation summary includes the following structure (dict):
```Python
eval_summary = {
    'error_app_cnt':error_app_cnt,
    'overall_error_app_cnt':overall_error_app_cnt,
    'error_app_ratio':error_app_ratio,
    'overall_error_app_ratio':overall_error_app_ratio,
    'error_content_cnt':error_content_cnt,
    'overall_error_content_cnt':overall_error_content_cnt,
    'error_content_ratio':error_content_ratio,
    'overall_error_content_ratio':overall_error_content_ratio,
    'error_logic_cnt':error_logic_cnt,
    'overall_error_logic_cnt':overall_error_logic_cnt,
    'error_logic_ratio':error_logic_ratio,
    'overall_error_logic_ratio':overall_error_logic_ratio,
    'score_app':score_app,
    'score_content':score_content,
    'score_logic':score_logic
}
```
Using the evaluation summary below, the stakeholders might decide the threshold of a *good* response that does not need to go through the improvement process.

This process **does not** include the improvement process.

## Improvement
This part is for improving an existing response using our taxonomy and the feedbacks if given.

```ruby
python improve.py --response_dir "data/responses/" \
                  --response_file "response_file" \
                  --improve_dir "data/improved/" \
                  --improve_file "improve_file" \
                  --question_col "question" \
                  --id_col "guid" \
                  --response_col "numbered_response" \
                  --need_eval "True/False" \
                  --mode "score/error/other_name_you_want" \
                  --include_short_def "True/False" \
                  --include_def "True/False" \
                  --model_name "GPT-4o" \
                  --model_path "gpt-4o-2024-05-13" \
                  --temperature 1 \
                  --top_p 0.9 \
                  --is_pretrained "False" \
                  --gpt_azure "False"
```

By running the code above, the responses in `response_dir/response_file`'s `response_col` is improved using the evaluation results of itself. Setting `need_eval` as `"True"` will run the evaluation code on the original response too, meaning that there is no need to run the evaluation code from the previous section. However, if you want to use the existing feedback, set `response_dir/response_file` as the file with the evaluation results from the previous section and set `need_eval` as `"False"`. 

Setting `include_short_def` as `"True"` will include the short and simple description of a good response based on our taxonomy (*"[개선된 응답]은 주어진 질문의 맥락에 맞게 명확하게 답하며, 다양한 인간적 가치와 의견을 존중하며 잠재적으로 유해한 내용이 없어야 한다. 이와 동시에 논리적 오류 및 비효율이 없어야 한다."*) within the improvement prompt. `include_def` decides whether to include the whole definition of the taxonomy within the improvement prompt or not.

Running the code below will use the evaluation results if you had run `bash evaluation.sh`.
```ruby
$ bash improve.sh
```

Running the code below will perform evaluation and improvement of the given responses, with no need of already-evalauted responses.
```ruby
$ bash improve_from_scratch.sh
```
You can also run improvement on a single question & response pair by using the `improve_single_response()` function within `improve.py`.
