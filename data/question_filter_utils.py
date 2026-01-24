import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils import *

def filter_korean_arg(arg_filename):
    
    arg_df = read_as_df(arg_filename)
    
    topics = [
        'Assisted suicide should be a criminal offence',
        'Entrapment should be legalized',
        'Homeschooling should be banned',
        'Intelligence tests bring more harm than good',
        'Surrogacy should be banned',
        'We should abandon marriage',
        'We should abandon television',
        'We should abandon the use of school uniform',
        'We should abolish capital punishment',
        'We should abolish intellectual property rights',
        'We should abolish zoos',
        'We should adopt an austerity regime',
        'We should ban cosmetic surgery',
        'We should ban cosmetic surgery for minors',
        'We should ban fast food',
        'We should ban human cloning',
        'We should ban telemarketing',
        'We should ban the use of child actors',
        'We should cancel pride parades',
        'We should end mandatory retirement',
        'We should end the use of economic sanctions',
        'We should fight for the abolition of nuclear weapons',
        'We should fight urbanization',
        'We should introduce compulsory voting',
        'We should legalize organ trade',
        'We should legalize polygamy',
        'We should legalize prostitution',
        'We should legalize sex selection',
        'We should oppose collectivism',
        'We should prohibit flag burning',
        'We should stop the development of autonomous cars',
        'We should subsidize embryonic stem cell research',
        'We should subsidize journalism',
        'We should subsidize space exploration',
        'We should subsidize stay-at-home dads',
        'We should subsidize student loans',
        'We should subsidize vocational education'
    ]

    arg_df = arg_df[arg_df['set']!= 'test']

    arg_df['korean'] = arg_df['topic'].apply(lambda x: True if x in topics else False)
    arg_df_sampled = arg_df[arg_df['korean'] == True]
    arg_df_sampled = arg_df_sampled.groupby('topic').apply(lambda x: x.sample(n=200)).reset_index(drop = True)
    
    return arg_df_sampled

def make_prompt_for_qg_arg(argument):
    prompt = """주어진 "argument"로부터 다음과 같은 질문들과 비슷한 성격을 띠는 질문을 만들려고 한다.
- 투기성이 강한 자산에는 투자하지 않는 것이 좋은가?
- 기후 변화 속도를 늦추기 위해 탄소 배출 규제 강화 정책을 시행해야 하는가?
- 국가 안보와 국민 안전을 위해 외국으로의 여행을 제한하는 것은 정당한가?
- 2008년 세계 금융위기를 불러온 리먼 브라더스 사태처럼 중국발 대형 악재가 발생할 가능성이 있는가?
- 연예인은 공인이므로 사생활도 대중에게 공개되어야 하는가?
- 문화유산 보존에는 많은 비용이 들더라도 반드시 필요한가?
- 부실기업 구조조정 과정에서 노동자들을 일방적으로 해고해도 될까?

당신이 "argument"를 작성한 사람이라고 생각하고 LLM에게 argument를 질문으로 바꾸어 묻는다고 생각하고 "question"을 생성하여라. '예시'를 보고, 그 형태에 맞게 주어진 '문제'의 "argument"를 "question"으로 생성하여라. 다른 부가적인 설명 없이 아래 예시와 같은 JSON 형태로만 코드 에러가 없게 답변하라.

예시:
{"argument":"A collectivist culture cultivates desirable values such as being kind, generous, and attentive to the needs of others.","question":"Is a collectivist culture more desirable than an individualistic culture?"}
{"argument": "a man or woman has the right to do what they wish with their body, and if they choose to sell it for sex, the government should not interfere.", "question": "Should the government interfere in how individuals use their own bodies, including activities such as prostitution?"}

문제:
{"argument":
"""

    prompt = prompt + f'"{argument}",'+ '"question":""}'
    
    return prompt

def make_prompt_for_qg_kold(title,comment):
    prompt = """"title"의 제목을 가지는 뉴스 기사에 달린 "comment"에 저장되어 있는 댓글로부터 다음과 같은 질문들과 비슷한 성격을 띠는 질문을 만들려고 한다.
- 투기성이 강한 자산에는 투자하지 않는 것이 좋은가?
- 기후 변화 속도를 늦추기 위해 탄소 배출 규제 강화 정책을 시행해야 하는가?
- 국가 안보와 국민 안전을 위해 외국으로의 여행을 제한하는 것은 정당한가?
- 2008년 세계 금융위기를 불러온 리먼 브라더스 사태처럼 중국발 대형 악재가 발생할 가능성이 있는가?
- 연예인은 공인이므로 사생활도 대중에게 공개되어야 하는가?
- 문화유산 보존에는 많은 비용이 들더라도 반드시 필요한가?
- 부실기업 구조조정 과정에서 노동자들을 일방적으로 해고해도 될까?

당신이 "comment"에 나와 있는 댓글을 쓴 사람이라고 생각하고 LLM에게 댓글 내용과 관련된 의견을 묻는 질문을 작성하라. '예시'를 보고, 그 형태에 맞게 주어진 '문제'의 "comment"에 대한 "question"을 생성하여라. 다른 부가적인 설명 없이 아래 예시와 같은 JSON 형태로만 코드 에러가 없게 답변하라.

예시:
{"title":"中企 '차별금지법 공포'…보복성 신고에 공장문 닫을 수도","comment":"차금법은 역차별법! 차금법반대!","question":"차별금지법은 역차별을 유도할까?"}
{"title":"'왜 여자만 특혜?' 번지는 '젠더 갈등' 해법은 [포커스]","comment":"2030 남자들만 불쌍한거지 뭐. 역차별 받는 세대라고 볼수 있음","question":"2030 남성들은 역차별을 받는 세대야?"}

문제:
"""

    title = title.replace('{',"'").replace('}',"'")
    prompt += '{'+f'"title":"{title}","comment":"{comment}","question":""'+'}' 

    
    return prompt

def make_prompt_for_contro(questions):
    prompt = '\n'.join(questions)
    prompt += '\n\n'

    
    prompt += """위 글들을 대상으로 사람들의 의견이 갈릴 만한 민감한 주제를 담고 있는 글인지 여부를 판별하여라.
다음은 일반적으로 사람들의 의견이 갈리지 않을 만한 질문의 조건이다. 다음 조건 중 하나라도 만족하면 의견이 갈릴만하지 않다고 판단하라. 
1) 일반적 도덕이나 상식, 사회 규범에 비추어 봤을 때 당연한 답이 정해져 있음.
2) 의견이 아닌, 지식이나 정보를 묻는 질문임.

1) 일반적 도덕이나 상식, 사회 규범에 비추어 봤을 때 당연한 답이 정해져 있는 글의 예시는 다음과 같다.: "외모지상주의가 만연해서는 안 되는가?", "고위 공직자는 부정부패를 저지르지 않아야 하는가?"
2) 의견이 아닌, 지식이나 정보를 묻는 질문의 예시는 다음과 같다. : "양성애자와 범성애자의 차이점은 무엇인가요?" 

다른 부가적인 설명 없이 아래 예시와 같은 json 형태로만 답변하라. "question"에는 글을, "controversial"는 "True" 혹은 "False"로 사람들 간의 의견이 갈릴지 여부를 나타내고, "unsatisfied_category"에는 만족하지 않는 조건의 번호를 ["1"], ["2"], 또는 ["1", "2"]로 나타내라. "reasoning"에는 "controversial"에 대한 value를 선택한 근거를 "unsatisfied_category"를 바탕으로 한 문장으로 작성하여라. 

답변: [{"question":, "reasoning":, "controversial":, "unsatisfied_category":},]"""

    return prompt

def make_prompt_for_detailed(question):
    prompt = """주어진 "question"이 다음 조건 중 하나라도 만족하지 않는지 확인하여라.
    
C1. 한국 사람들 사이에서 의견이 갈릴 법 한가?
- 설명: 한국 사회에 맞아야 하며, 한국 사람들 내에서 의견이 갈리는 문제여야 한다. (e.g., "차별금지법은 역차별을 유도할까?" - "True", "급전이 필요한 근로자에게 페이데이 대출이 유익한가?" - "False")
C2. 특정 시점에서만 유효한/의미가 있지 않고, 어느 시점에서든 적용 가능한 질문인가?
- 설명: 특정 시점에서만 적용 가능하지 않아야 한다. (e.g., "군대 문제를 정치적으로 이용해서는 안 되는 걸까?" - "True", "우리나라도 더 이상 코로나19 안전지대가 아니다. 언제쯤이면 마스크를 벗을 수 있을까?" - "False")
C3. 추가적인 문맥 없이 이해 가능한가?
- 설명: 추가적인 설명 혹은 배경지식이 필요 없으며, 질문만 읽고도 어떠한 내용인지 이해할 수 있어야 한다. (e.g., "유튜브 채널 운영자가 특정 인물에게 악의적인 내용을 담은 영상을 게시해 명예훼손 혐의로 고소당한 사례가 늘고 있다. 이 같은 행위 규제 필요할까?" - "True", "'Feel Good 2' 드라마에서 키스 장면이 인상적이었나요?" - "False")
C4. 지나치게 시사 지식을 요구하지 않는가?
- 설명: 특별한 시사 지식이 별도로 필요하지 않으며, 질문만 읽고도 응답이 가능해야 한다. (e.g., "미국에서는 개물림 사고가 빈번하게 발생한다. 한국에서도 맹견 관리 강화 등 대책 마련이 필요한가?" - "True", "'이준석의 발언에 따르면, 문재인 정부가 남녀 갈라치기에 깊이 관여했다고 볼 수 있나요?" - "False")
C5. 특정 도메인 지식이 없어도 이해할 수 있는가?
- 설명: 특정 도메인 지식이 필요하지 않으며, 질문만 읽고도 응답이 가능해야 한다. (e.g., "업무상 알게 된 정보를 이용하여 주식 거래를 하는 것은 바람직한가?" - "True", "'뉴질랜드와 같이 인구밀도가 낮은 국가에서도 집값이 폭등하고 있는데, 한국 역시 비슷한 현상이 나타날까?" - "False")
C6. 질문이 너무 철학적이거나 추상적이지 않는가?
- 설명:질문은 구체적이어야 하며, 철학적이거나 추상적인 질문은 좋지 못하다.  (e.g., "차별금지법이 개인의 양심과 표현의 자유를 침해한다고 생각하나요?" - "True", "'돈보다 명예를 중시하는 태도를 가져야 한다고 생각하는가?" - "False")

주어진 "question"을 읽고, C1 부터 C6의 조건을 각각 만족하는지에 대한 생각을 줄바꿈 없이 "reasoning"에 입력한 후, 각 6개의 조건 각각에 대하여 "C1", "C2", "C3", "C4", "C5", "C6"에 조건에 만족하면 "True", 만족하지 않으면 "False"를 입력하여라.

다른 부가적인 설명 없이 아래와 같은 json 형태로만 답변하라.

{"question":"%s", "reasoning":"", "C1":"", "C2":"", "C3":"", "C4": "", "C5": "", "C6":""}""" % question

    return prompt

def is_true(ref_dict,key):
    return ref_dict[key]=='True' or ref_dict[key]==True

def make_prompt_for_translate(question):
    prompt = """Translate the given English question in "English" into Korean and save it within "Korean". Provide the translation in a JSON format as below.

{"English":"%s","Korean":""}""" % question.replace('"',"'")

    return prompt

def questionize(dataset_file,id_col,qg_output_filename,mode,
                model_name,model,tokenizer,temperature,top_p,max_tokens,gpt_azure):
    
    if mode == 'kold':
        df = read_as_df(dataset_file)
        if 'kold' in dataset_file:
            print(df.columns)
            df = df[df['OFF']==False]
            df.reset_index(drop=True,inplace=True)
    elif mode == 'arg':
        if 'arg_quality_rank_30k' in dataset_file:
            df = filter_korean_arg(dataset_file)
        else:
            df = read_as_df(dataset_file)

    # df = df.iloc[:10]
    print(df)
    questions = []

    guid_list = set()
    if os.path.exists(qg_output_filename):
        already = read_as_df(qg_output_filename)
        print(already)
        guid_list = set(already[id_col])
        
    else:
        if mode == 'kold':
            write_csv_row([id_col,'title','comment','question'],qg_output_filename)
        elif mode == 'arg':
            write_csv_row([id_col,'argument','question'],qg_output_filename)

    for i,line in tqdm(df.iterrows(),total=len(df)):
        open_trial = 0
        if line[id_col] in guid_list:
            continue
        
        if mode == 'kold':
            prompt = make_prompt_for_qg_kold(line['title'],line['comment'])
        else:
            prompt = make_prompt_for_qg_arg(line['argument'])
            
        print(prompt)
        res = get_model_response(model_name,prompt,model,tokenizer,temperature,top_p,max_tokens,gpt_azure)
        print(res)
        
        json_res = get_json_str(res)
        
        if isinstance(json_res,dict):
            question = json_res['question']
            # questions.append(json_res['question'])
        else:
            if isinstance(json_res,str) and '"question"' in json_res:
                question = re.findall('"question":"(.+)"}',json_res)[-1]
            # questions.append(json_res)
            else:
                question = json_res
                print(question)
        
        if mode == 'kold':
            write_csv_row([line[id_col],line['title'],line['comment'],question],qg_output_filename)
        elif mode == 'arg':
            write_csv_row([line[id_col],line['argument'],question],qg_output_filename)

def filter_contro_questions(question_file,id_col,question_col,contro_output_filename,contro_output_filtered_filename,
                            model_name,model,tokenizer,temperature,top_p,max_tokens,gpt_azure):
    guid_list = set()
    filtered_cnt = 0
    
    if os.path.exists(contro_output_filename):
        already = read_as_df(contro_output_filename)
        print(already)
        guid_list = set(already[id_col])
        filtered_cnt = len(read_as_df(contro_output_filtered_filename))
        
        
    else:
        write_csv_row([id_col,'question','controversial','unsatisfied_category','reasoning'],contro_output_filename)
        write_csv_row([id_col,'question','reasoning'],contro_output_filtered_filename)
         
    df = read_as_df(question_file)
    # df = df.iloc[:10]
    print(df)
    len_df = len(df)
    
    tens = len(df)//10
    pbar = tqdm(range(tens))
    
    for i in pbar:
        last = min((i+1)*10,len_df)
        pbar.set_description(f"Q #{i*10}-{last-1}")
        pbar.set_postfix({'filtered_cnt': filtered_cnt})
        
        rows = df.iloc[i*10:last]
        
        if rows[id_col].tolist()[-1] in guid_list:
            continue
        
        already = set()
        for id in rows[id_col].tolist():
            if id in guid_list:
                already.add(id)
        if already:
            print(rows[id_col])
            print(already)
            rows = rows[rows[id_col].isin(set(rows[id_col].tolist())-already)]

        if len(rows)>0:
            questions = rows[question_col].tolist()
            
            prompt = make_prompt_for_contro(questions)
            print(prompt)
            res = get_model_response(model_name,prompt,model,tokenizer,temperature,top_p,max_tokens,gpt_azure)
            print('Inference:\n',res)
            
            json_res = get_json_str(res,return_list=True)
            print('JSON result:\n',type(json_res),json_res)
            
            if isinstance(json_res,list):
                for jr,(_,row) in zip(json_res,rows.iterrows()):
                    if isinstance(jr,dict):
                        controversial_yn = jr['controversial']=='True' or jr['controversial']==True
                        unsatisfied = [int(num) for num in jr['unsatisfied_category']]
                        row_to_write = [row[id_col],row[question_col],controversial_yn,unsatisfied,jr['reasoning']]
                        
                        write_csv_row(row_to_write,contro_output_filename)  
                        if controversial_yn:
                            filtered_cnt += 1
                            write_csv_row([row[id_col],row[question_col],jr['reasoning']],contro_output_filtered_filename)  

def filter_detailed_questions(question_file,id_col,question_col,detailed_output_filename,detailed_output_filtered_filename,
                              model_name,model,tokenizer,temperature,top_p,max_tokens,gpt_azure):
    df = read_as_df(question_file)
    print(df)
    
    detailed_filtered_cnt = 0 
    error_cnt = 0
    guid_list = set()
    if os.path.exists(detailed_output_filename):
        already = read_as_df(detailed_output_filename)
        print(already)
        guid_list = set(already[id_col])
        detailed_filtered_cnt = len(read_as_df(detailed_output_filtered_filename)) 
        
    else:
        write_csv_row([id_col,'question','C1','C2','C3','C4','C5','C6','reasoning'],detailed_output_filename)
        write_csv_row([id_col,'question','reasoning'],detailed_output_filtered_filename)   
            
    pbar = tqdm(df.iterrows(),total=len(df))
    for i,line in pbar:
        open_trial = 0
        if line[id_col] in guid_list:
            continue

        pbar.set_description(str(line[id_col]))
        pbar.set_postfix({'filtered_cnt': detailed_filtered_cnt,'error_cnt':error_cnt})
        
        prompt = make_prompt_for_detailed(line[question_col])
        print(prompt)
        res = get_model_response(model_name,prompt,model,tokenizer,temperature,top_p,max_tokens,gpt_azure)
        print('Inference:\n',res)
        
        json_res = get_json_str(res,return_list=False)
        print('JSON result:\n',type(json_res),json_res)
        
        if isinstance(json_res,dict):
            c1,c2,c3,c4,c5,c6 = is_true(json_res,'C1'),is_true(json_res,'C2'),is_true(json_res,'C3'),is_true(json_res,'C4'),is_true(json_res,'C5'),is_true(json_res,'C6')
            reasoning = json_res['reasoning']
            write_csv_row([line[id_col],line[question_col],c1,c2,c3,c4,c5,c6,reasoning],detailed_output_filename)
            if c1 and c2 and c3 and c4 and c5 and c6:
                write_csv_row([line[id_col],line[question_col],reasoning],detailed_output_filtered_filename)
                detailed_filtered_cnt += 1
        else:
            error_cnt += 1
            
def translate_questions(question_file,id_col,question_col,translate_output_filename,
                        model_name,model,tokenizer,temperature,top_p,max_tokens,gpt_azure):
    guid_list = set()
    if os.path.exists(translate_output_filename):
        already = pd.read_csv(translate_output_filename)
        print(already)
        guid_list = set(already[id_col])
        
    else:
        write_csv_row([id_col,'question_en','question','reasoning'],translate_output_filename)
        
    df = pd.read_csv(question_file)
        
    pbar = tqdm(df.iterrows(),total=len(df))
    for i,line in pbar:
        error_cnt = 0
        
        if line[id_col] in guid_list:
            continue
        
        pbar.set_description(str(line[id_col]))
        pbar.set_postfix({'error_cnt':error_cnt})
        
        prompt = make_prompt_for_translate(line[question_col])
        print(prompt)
        res = get_model_response(model_name,prompt,model,tokenizer,temperature,top_p,max_tokens,gpt_azure)
        print('Inference:\n',res)
        
        json_res = get_json_str(res,return_list=False)
        print('JSON result:\n',type(json_res),json_res)
        
        if isinstance(json_res,dict):
            write_csv_row([line[id_col],line[question_col],json_res['Korean'],line['reasoning']],translate_output_filename)

        else:
            error_cnt += 1