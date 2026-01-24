from generate_responses_utils import *

def post_process():
    response_file = os.path.join(args.response_dir,args.response_file)
    numbered_response_file = os.path.join(args.response_dir,args.numbered_response_file)
    
    tokenizer,model = get_tokenizer_model(args.model_name,args.model_path,args.model_cache_dir,args.is_pretrained)
    
    number_sents(response_file,args.id_col,args.question_col,numbered_response_file,
                 args.model_name,args.model_path,model,tokenizer,args.temperature,args.top_p,args.max_tokens,args.gpt_azure)
    
    filter_responses(numbered_response_file,args.id_col)

if __name__ == '__main__':
    post_process()