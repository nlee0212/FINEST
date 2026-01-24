from generate_responses_utils import *

def get_responses():
    question_file = os.path.join(args.question_dir,args.question_file)
    if not os.path.exists(args.response_dir):
        os.mkdir(args.response_dir)
    response_file = os.path.join(args.response_dir,args.response_file)
    
    tokenizer,model = get_tokenizer_model(args.model_name,args.model_path,args.model_cache_dir,args.is_pretrained)
    
    generate_response(question_file,args.id_col,args.question_col,args.dataset,response_file,
                      args.model_name,args.model_path,model,tokenizer,args.temperature,args.top_p,args.max_tokens,args.gpt_azure)

if __name__ == '__main__':
    get_responses()