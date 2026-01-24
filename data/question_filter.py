from question_filter_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir',type=str,default=None,
                    help='Provide the directory name with the datasets.')
parser.add_argument('--raw_dataset_file',type=str,default=None,
                    help='Provide the dataset file name to be questionized or filtered. Only csv, json, jsonl files are available.')
parser.add_argument('--question_col',type=str,default=None,
                    help='Provide the column/key name from the given dataset file name with questions.')
parser.add_argument('--id_col',type=str,default=None,
                    help='Provide the column/key name from the given dataset file name with question IDs.')
parser.add_argument("--need_translation", type=str2bool, nargs='?',
                    const=True, default=False,
                    help="Whether you need to translate questions from English to Korean.")

parser.add_argument('--model_name',type=str,
                    help='Provide the name of the model you want to use. This is only for you to keep track of it.')
parser.add_argument('--model_path',type=str,
                    help='Provide the path of the model you want to use. This will be used when running inference for APIs or download the pretrained models from huggingface.')
parser.add_argument('--model_cache_dir',type=str,default='.cache',
                    help='Provide the directory saving model caches.')
parser.add_argument('--temperature',type=float,default=0,
                    help='Provide generation temperature for models.')
parser.add_argument('--top_p',type=float,default=0,
                    help='Provide generation top_p for models.')
parser.add_argument('--max_tokens',type=int,default=None,
                    help='Provide generation max tokens for models.')
parser.add_argument("--is_pretrained", type=str2bool, nargs='?',
                    const=True, default=False,
                    help="Whether you are using the pretrained models for response generation.")
parser.add_argument("--gpt_azure", type=str2bool, nargs='?',
                    const=True, default=False,
                    help="Whether you are using the AzureOpenAI for GPT-models' response generation.")

parser.add_argument("--questionize_mode", type=str,default=None,
                    help="Choose between 'kold' or 'arg' if you want to questionize the given dataset file. 'kold' uses title & comment, and 'arg' uses the argument.")


args = parser.parse_args()

def filter_question():
    
    tokenizer,model = get_tokenizer_model(args.model_name,args.model_path,args.model_cache_dir,args.is_pretrained)
    file_type = '.'+args.raw_dataset_file.split('.')[-1]
    
    if not os.path.exists('processed'):
        os.mkdir('processed')
    if not os.path.exists('final_questions'):
        os.mkdir('final_questions')
    
    start_file = os.path.join(args.dataset_dir,args.raw_dataset_file)
    questionized_filename = 'processed/'+args.raw_dataset_file.replace(file_type,'_questionized.csv')
    contro_filename = 'processed/'+args.raw_dataset_file.replace(file_type,'_contro.csv')
    contro_filtered_filename = 'processed/'+args.raw_dataset_file.replace(file_type,'_contro_filtered.csv')
    detailed_filename = 'processed/'+args.raw_dataset_file.replace(file_type,'_detailed.csv')
    detailed_filtered_filename = 'processed/'+args.raw_dataset_file.replace(file_type,'_detailed_filtered.csv')
    translated_filename = 'processed/'+args.raw_dataset_file.replace(file_type,'_translated.csv')
    final_filename = 'final_questions/'+args.raw_dataset_file.replace(file_type,'_final.csv')
    question_col = args.question_col
    
    if args.id_col == None:
        df = read_as_df(start_file)
        args.id_col = 'guid'
        df[args.id_col] = list(range(len(df)))
        start_file = start_file.replace(file_type,'.csv')
        df.to_csv(start_file,index=False,encoding='utf-8')
    
    if args.questionize_mode in ['kold','arg']:
        if args.questionize_mode == 'kold':
            questionize(start_file,args.id_col,questionized_filename,'kold',
                        args.model_path,model,tokenizer,args.temperature,args.top_p,args.max_tokens,args.gpt_azure)

        elif args.questionize_mode == 'arg':
            questionize(start_file,args.id_col,questionized_filename,'arg',
                        args.model_path,model,tokenizer,args.temperature,args.top_p,args.max_tokens,args.gpt_azure)
        
        start_file = questionized_filename
        question_col = 'question'
    
    filter_contro_questions(start_file,args.id_col,question_col,contro_filename,contro_filtered_filename,
                            args.model_path,model,tokenizer,args.temperature,args.top_p,args.max_tokens,args.gpt_azure)
    
    filter_detailed_questions(contro_filtered_filename,args.id_col,'question',detailed_filename,detailed_filtered_filename,
                              args.model_path,model,tokenizer,args.temperature,args.top_p,args.max_tokens,args.gpt_azure)
    shutil.copyfile(detailed_filtered_filename, final_filename)
    
    if args.need_translation:
        translate_questions(detailed_filtered_filename,args.id_col,'question',translated_filename,
                            args.model_path,model,tokenizer,args.temperature,args.top_p,args.max_tokens,args.gpt_azure)
    
        shutil.copyfile(translated_filename, final_filename)

if __name__ == '__main__':
    filter_question()