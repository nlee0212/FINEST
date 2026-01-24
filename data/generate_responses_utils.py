import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--question_dir',type=str,default=None,
                    help='Provide the directory name with the datasets.')
parser.add_argument('--question_file',type=str,default=None,
                    help='Provide the dataset file name with the questions. Only csv, json, jsonl files are available.')
parser.add_argument('--response_dir',type=str,default=None,
                    help='Provide the directory name with the datasets.')
parser.add_argument('--response_file',type=str,default=None,
                    help='Provide the dataset file name to store the responses. Only csv, json, jsonl files are available.')
parser.add_argument('--numbered_response_file',type=str,default=None,
                    help='Provide the dataset file name to store the numbered responses. Only csv, json, jsonl files are available.')
parser.add_argument('--question_col',type=str,default=None,
                    help='Provide the column name from the given dataset file name with questions.')
parser.add_argument('--id_col',type=str,default=None,
                    help='Provide the column name from the given dataset file name with question IDs.')
parser.add_argument('--dataset',type=str,default=None,
                    help='Provide the dataset name for the given dataset file name. This is only for you to keep track of it. But DO NOT use "." within the dataset name.')

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

args = parser.parse_args()

def make_prompt(question,yes_no=None):
    if yes_no == True:
        prompt = "'%s'에 대하여 '그렇다'고 생각하는 입장에서 답변을 작성하시오. 다만 자신이 어떠한 입장인지 명시하지 말고, 해당 프롬프트가 사용자에게 보이지 않도록 위 의견을 가지고 있는 사람이 할 답변만 작성하시오." % question
    elif yes_no == False:
        prompt = "'%s'에 대하여 '그렇지 않다'고 생각하는 입장에서 답변을 작성하시오. 다만 자신이 어떠한 입장인지 명시하지 말고, 해당 프롬프트가 사용자에게 보이지 않도록 위 의견을 가지고 있는 사람이 할 답변만 작성하시오." % question 
    else:
        prompt = "%s" % question
    
    return prompt

def generate_response(question_file,id_col,question_col,dataset_name,response_file,
                      model_name,model_path,model,tokenizer,temperature,top_p,max_tokens,gpt_azure):
    
    data = read_as_df(question_file)
    
    guid_list = set()
    if os.path.exists(response_file):
        already = pd.read_csv(response_file)
        print(already)
        guid_list = set(already[id_col])
        
    else:        
        write_csv_row([id_col,question_col,'answer_type','model','response'],response_file)
    
    pb = tqdm(data.iterrows(),total=len(data))
    for _,d in pb:
        already_res = [False,False,False]
        q = d[question_col]
        guid = d[id_col]
        pb.set_description(str(guid))
        
        if guid in guid_list:
            already_row = len(already[already[id_col]==guid])
            for i in range(already_row):
                already_res[i] = True
        
        if not already_res[0]:
            yes_prompt = make_prompt(q,True)
            print(yes_prompt)
            yes_res = get_model_response(model_path,yes_prompt,model,tokenizer,temperature,top_p,max_tokens,gpt_azure)
            print(yes_res)
            write_csv_row([guid,q,'Yes',model_name,yes_res],response_file) 
        if not already_res[1]:
            no_prompt = make_prompt(q,False)
            print(no_prompt)
            no_res = get_model_response(model_path,no_prompt,model,tokenizer,temperature,top_p,max_tokens,gpt_azure)
            print(no_res)
            write_csv_row([guid,q,'No',model_name,no_res],response_file) 
        if not already_res[2]:    
            general_prompt = make_prompt(q,None)
            print(general_prompt)
            general_res = get_model_response(model_path,general_prompt,model,tokenizer,temperature,top_p,max_tokens,gpt_azure)
            print(general_res)
            write_csv_row([guid,q,'General',model_name,general_res],response_file)

def number_sents(response_file,id_col,question_col,output_filename,
                 model_name,model_path,model,tokenizer,temperature,top_p,max_tokens,gpt_azure):
    
    data = pd.read_csv(response_file,encoding='utf8')
    
    guid_list = set()
    if os.path.exists(output_filename):
        already = pd.read_csv(output_filename)
        print(already)
        guid_list = set(already[id_col])
    else:
        write_csv_row([id_col,question_col,'answer_type','model','response','numbered_response'],output_filename)
        
    pb = tqdm(data.iterrows(),total=len(data))
    for _,d in pb:
        r = d['response']
        guid = d[id_col]
        pb.set_description(str(guid))
        pb.set_postfix({'file_name': response_file})
        
        if guid in guid_list and (d['answer_type'] in list(already[already[id_col]==guid]['answer_type'].astype(str))):
            continue
        
        prompt = make_prompt_number_sents(r)
        response = get_model_response(model_path,prompt,model,tokenizer,temperature,top_p,max_tokens,gpt_azure)
        print(response)
        write_csv_row([guid,d[question_col],d['answer_type'],d['model'],r,response],output_filename)
        
def check_string(s):
    # This regular expression looks for any newline
    pattern = r'\n+'
    
    # This regular expression looks for a newline followed by '[\d+]'
    desired_pattern = r'\n+\[\d+\]'
    
    # Find all newlines
    all_newlines = re.findall(pattern, s)
    
    # Find all newlines that are correctly followed by '[\d+]'
    correct_newlines = re.findall(desired_pattern, s)
    # print(correct_newlines)
    # Check if all occurrences of newlines are followed by '[\d+]'
    return len(all_newlines) == len(correct_newlines)

def filter_responses(response_file,id_col):
    
    df = read_as_df(response_file)
    
    df['numbered_response'] = df['numbered_response'].astype(str)
    df = df.dropna()

    tmp1 = df[~df['numbered_response'].str.startswith('[1]')]
    tmp2 = df[df['numbered_response'].str.contains('OTHER')]
    tmp3 = df[df['response']=='content_filter']
    tmp4 = df[df['response']=='답변 오류 발생']
    tmp5 = df[df['response']=='답변 할 수 없습니다.']
    tmp6 = df[~(df['numbered_response'].str.contains('[2]')) & ~(df['numbered_response'].str.endswith('.'))]
    tmp7 = df[df['numbered_response'].str.contains(', \[\d+\]')]
    tmp8 = df[~df['numbered_response'].apply(check_string)]
    tmp9 = df[(df['numbered_response'].str.contains('\D\\.\s+')) & ~(df['numbered_response'].str.contains('\D\\.\s+\[\d+\]'))]
    
    concat_tmp = pd.concat([tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7,tmp8,tmp9])

    filtered_df = df[~df.index.isin(concat_tmp.index)]
    
    answer_type_categories = ['Yes',"No",'General']

    filtered_df['answer_type'] = pd.Categorical(df['answer_type'],categories=answer_type_categories)
    filtered_df = filtered_df.sort_values(by=[id_col,'answer_type'])
    filtered_df.reset_index(drop=True,inplace=True)
    
    filtered_df.to_csv(response_file,encoding='utf-8',index=False)