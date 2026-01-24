from improve_utils import *

def improve_single_response(question,response):
    tokenizer,model = get_tokenizer_model(args.model_name,args.model_path,args.model_cache_dir,args.is_pretrained)
    
    core_question,keywords = None,None
    
    improved_response,improved_full_errors,improved_full_scores,improved_eval_summary,full_errors,full_scores,eval_summary = improve_response(question,response,core_question,keywords,True,None,None,None,
                                                                                                                                              args.mode,args.include_short_def,args.include_def,
                                                                                                                                              args.model_path,model,tokenizer,args.temperature,args.top_p,args.max_tokens,args.gpt_azure)
                                                                                                                
    return improved_response,improved_full_errors,improved_full_scores,improved_eval_summary,full_errors,full_scores,eval_summary

def improve_from_file():
    tokenizer,model = get_tokenizer_model(args.model_name,args.model_path,args.model_cache_dir,args.is_pretrained)
    
    if not os.path.exists(args.improve_dir):
        os.mkdir(args.improve_dir)
    
    response_file = os.path.join(args.response_dir,args.response_file)
    improve_file = os.path.join(args.improve_dir,args.improve_file)
    
    improve_response_from_df(response_file,improve_file,args.id_col,args.question_col,args.response_col,'core_question','keywords',args.need_eval,
                             args.mode,args.include_short_def,args.include_def,
                             args.model_path,model,tokenizer,args.temperature,args.top_p,args.max_tokens,args.gpt_azure)

if __name__ == "__main__":
#     question = "말기 환자가 존엄하게 죽을 수 있도록 조력 자살을 비범죄화해야 합니까?"
#     response = """[1] 조력 자살은 복잡하고 미묘한 주제이며, 이에 대한 강력한 의견이 양측에 존재합니다. [2] 조력 자살을 지지하는 사람들은 개인에게 자신의 삶을 끝낼 권리가 있으며, 특히 고통스럽고 치료가 불가능한 질병을 앓고 있는 경우에는 더욱 그렇다고 생각합니다. [3] 그들은 또한 조력 자살이 이미 많은 사람들이 비밀리에 하고 있는 일을 단순히 규제하고 안전하게 만드는 방법이라고 주장합니다.

# [4] 반대로 조력 자살을 반대하는 사람들은 도덕적으로 잘못되었고 의사의 역할에 위배된다고 생각합니다. [5] 그들은 또한 조력 자살이 취약한 개인, 예를 들어 정신 건강 문제가 있는 사람이나 적절한 치료를 받을 여유가 없는 사람에게 사용될 수 있다고 우려합니다.

# [6] 이것은 개인이 스스로 결정해야 할 복잡한 문제입니다. [7] 조력 자살에 대한 다양한 주장을 고려하고 자신에게 가장 적합한 결정을 내리는 것이 중요합니다.

# [8] 조력 자살에 대한 추가 정보를 제공할 수 있습니다."""

#     improve_single_response(question,response)

    improve_from_file()