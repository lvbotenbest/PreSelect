from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments,BitsAndBytesConfig
import torch
import os
import json
from tqdm import tqdm
import random
import argparse
import random
import time




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default="/home/zhangyanan/zyn/lb/ragmedge/llm-master/pretrain-model/Llama-2-7b-chat-hf", type=str,
                    help='model path')
    parser.add_argument('--test_file', default="./model_outputs.json", type=str,
                    help='output path')
    args = parser.parse_args()  
    return args


# def find_special_token_pos(input_ids,special_token_ids):

#     input_ids = input_ids.squeeze()
#     special_token_ids = special_token_ids.squeeze()

#     # 初始化结果列表
#     positions = []

#     # 滑动窗口搜索
#     for i in range(input_ids.shape[-1] - special_token_ids.shape[-1] + 1):
#         if torch.equal(input_ids[i:i + special_token_ids.shape[-1]], special_token_ids):
#             positions.append(i)

#     return positions

def find_special_token_pos(input_ids,special_token_ids):

    input_ids = input_ids.squeeze().tolist()
    special_token_ids = special_token_ids.squeeze().tolist()
    

    # 初始化结果列表
    positions = []

    special_len = len(special_token_ids)
    # 滑动窗口搜索
    for i in range(len(input_ids) - special_len + 1):
        if input_ids[i:i + special_len] ==special_token_ids:
            positions.append(i)

    return positions


def find_discrimate_logits(logits,special_token_ids,input_ids):

    pos_start = find_special_token_pos(input_ids,special_token_ids)[:-1]

    # pos_end = list(map(lambda x: x + special_token_ids.shape[-1], pos_start))

    label_list = []
    input_ids = input_ids.squeeze()
    bias = []
    true_bias = 0
    false_bias = 0

    num_false = 0
    num_true = 0

    num_bias = 0
    true_logits_sum = 0
    fasle_logits_sum = 0

    all_true = 0
    all_false = 0



    special_token_length = special_token_ids.shape[-1]

    pos_start = pos_start[-30:]
    for pos in pos_start:


        pos = pos+special_token_length
        label = input_ids[pos].item()
        label_list.append(label)
        label_score = logits[pos-1,label].item()
        if label == 3009:
            true_logits_sum += label_score
            all_true += 1
            false_ = 4541
        else:
            false_ = 3009
            fasle_logits_sum += label_score
            all_false += 1

        bias_ = label_score - logits[pos-1,false_].item()
        # bias.append(bias_)

        if bias_< 0:
            num_bias+=1
            if label == 3009:
                true_bias -= bias_
                num_true +=1
            else:
                false_bias -=bias_
                num_false+=1


    # print(pos_end)
    # print(label_list)
    # print(bias)
    # print(true_bias)
    # print(false_bias)
    
    # print("label is true predict is 'false':  ",num_true,end="       ")
    # print("label is false predict is 'true':  ",num_false)

    # if true_bias>false_bias:
    #     return True,(true_bias-false_bias)/len(pos_start)
    # else:
    #     return False,(false_bias-true_bias)/len(pos_start)

    # all_true = max(all_true,1)
    # all_false = max(all_false,1)

    # print(all_true)
    # print(all_false)
    # print(true_logits_sum/all_true,end="   ")
    # print(fasle_logits_sum/all_false)
    # return true_logits_sum/all_true,fasle_logits_sum/all_false
    
    if num_bias>0:

        num_true = max(num_true,1)
        num_false = max(num_false,1)

        return -true_bias/num_true,-false_bias/num_false
     
    else:
        return False,0


    # if num_bias>0:
    #     if true_bias>false_bias:
    #         return True,(true_bias-false_bias)/num_bias
    #     else:
    #         return False,(false_bias-true_bias)/num_bias
    # else:
    #     return False,0



def single_inference(input_text,model,tokenizer,max_token):
    messages = [{"role": "user", "content": input_text},]

    # input_ids = tokenizer.apply_chat_template(
    #                         messages,
    #                         add_generation_prompt=True,
    #                         return_tensors="pt"
    #                         ).to(model.device)
    
    input_ids = tokenizer.encode(input_text,return_tensors="pt").to(model.device)
    # print(input_ids)


    generate_kwargs = dict(
            input_ids=input_ids,
            do_sample=True,
            temperature=0.6, 
            top_p=0.9, 
            pad_token_id = tokenizer.eos_token_id,
            max_new_tokens=max_token,
            # return_dict_in_generate=True,
            # output_hidden_states=True,
            )
    outputs = model.generate(**generate_kwargs)
    return tokenizer.decode(outputs[0][input_ids.shape[-1]:],skip_special_tokens = True)



# def signficance


def single_inference_true_or_flase(input_text,model,tokenizer,max_token,add_bias):
 
    add_bias = False
    if not add_bias:
        input_ids = tokenizer.encode(input_text,return_tensors="pt").to(model.device)

        generate_kwargs = dict(
                input_ids=input_ids,
                do_sample=True,
                temperature=0.6, 
                top_p=0.9, 
                pad_token_id = tokenizer.eos_token_id,
                max_new_tokens=max_token,
                return_dict_in_generate=True,
                output_scores = True,
                output_logits=True,
                # output_hidden_states=True,
                )
        
        outputs = model.generate(**generate_kwargs)
        
        # true 3009 false 4541 

        # if outputs["scores"][0][0,4541].item()>0:
        #     return "false"

        if outputs["logits"][0][0,3009].item() > outputs["logits"][0][0,4541].item():
            return "true"
        else:
            return "false"
        

    else:
        input_ids = tokenizer.encode(input_text,return_tensors="pt")

        with torch.no_grad(): 
            outputs = model(input_ids.to(model.device), return_dict=True)
            end_token= torch.tensor([[ 13, 22550, 29901]])
            # dis,avg_bias = find_discrimate_logits(outputs["logits"][0],end_token,input_ids)

            prori_true,prori_false = find_discrimate_logits(outputs["logits"][0],end_token,input_ids)

            true_logits = outputs["logits"][0][-1,3009].item()-prori_true
            false_logits = outputs["logits"][0][-1,4541].item()-prori_false

        #if dis:
        #     true_logits = outputs["logits"][0][-1,3009].item()+avg_bias
        #     false_logits = outputs["logits"][0][-1,4541].item()
        # else:
        #     true_logits = outputs["logits"][0][-1,3009].item()
        #     false_logits = outputs["logits"][0][-1,4541].item()+avg_bias


        if true_logits > false_logits:
            return "true"
        else:
            return "false"



    return tokenizer.decode(outputs["sequences"][0][input_ids.shape[-1]:],skip_special_tokens = True)




def single_inference_true_or_flase_cad(wo_context,input_with_context,model,tokenizer):


    wo_context_input_ids = tokenizer.encode(wo_context,return_tensors="pt") 
    input_ids = tokenizer.encode(input_with_context,return_tensors="pt")

    
    with torch.no_grad(): 
        
        wo_context_outputs = model(wo_context_input_ids.to(model.device), return_dict=True)
        outputs = model(input_ids.to(model.device), return_dict=True)

        end_token= torch.tensor([[ 13, 22550, 29901]])

        wo_context_true_logits = wo_context_outputs["logits"][0][-1,3009].item()
        wo_context_false_logits = wo_context_outputs["logits"][0][-1,4541].item()

        with_context_true_logits = outputs["logits"][0][-1,3009].item()
        with_context_false_logits = outputs["logits"][0][-1,4541].item()

        true_logits = with_context_true_logits-wo_context_true_logits
        false_logits = with_context_false_logits-wo_context_false_logits


    if true_logits > false_logits:
        return "true"
    else:
        return "false"



    return tokenizer.decode(outputs["sequences"][0][input_ids.shape[-1]:],skip_special_tokens = True)






def demonstrate_instruct(output_file,sample_num):
    with open("./model_without_ret_acc/llama2_7b_not_ret.json", 'r') as f:
        data = json.load(f)



    demonstrate_list = []
    right_num = 0
    wrong_num = 0

    for item in data:
        if random.randint(1,100) == 34:
            if item["not_ret_answer_result"]=="true":
                if right_num<sample_num:
                    demonstrate_list.append(item)
                    right_num+=1
            else:
                if wrong_num<sample_num:
                    demonstrate_list.append(item)
                    wrong_num+=1
    
    random.shuffle(demonstrate_list)

    with open(output_file,"w") as file:
        json.dump(demonstrate_list, file,ensure_ascii=False, indent=4)


# demonstrate_instruct("./model_without_ret_acc/demonstrate_10.json",10)







def construct_instruction(demonsrate_file):
    instruction = "[INST] You are a student being tested. For each given question, assess based on your knowledge whether you can answer it correctly.\
 If you believe you can answer it correctly, output true. If you feel your answer might be incorrect, output false. \
\nHere are some examples and the output format.[/INST]\n\n"


    with open(demonsrate_file, 'r') as f:
        demonsrate_data = json.load(f)

    demonsrate_instruct = ""
    for line in demonsrate_data:
        question = line["question"]
        # answer = "true" if line["not_ret_answer_result"]== "false" else "false"
        # answer = "true" if random.randint(1,2)== 1 else "false"
        # answer = "false"
        answer = line["not_ret_answer_result"]
        one_demon_data = f"Question: {question}\nAnswer:{answer}\n"
        demonsrate_instruct+=one_demon_data

    return instruction+demonsrate_instruct


def only_instruction():
    instruction = "[INST] You are a student being tested. For each given question, assess based on your knowledge whether you can answer it correctly.\
 If you believe you can answer it correctly, output true. If you are unsure or need to reference external knowledge to answer it, output 'false'.[/INST]\n\n"

    return instruction



def construct_instruction_with_dynamic(demon_list):
    #v1
#     instruction = "[INST] You are a student being tested. For each given question, assess based on your knowledge whether you can answer it correctly.\
#  If you believe you can answer it correctly, output true. If you feel your answer might be incorrect, output false. \
# \nHere are some examples and the output format.[/INST]\n\n"

#v2
    instruction = "[INST] You are a student being tested. For each given question, assess based on your knowledge whether you can answer it correctly.\
 If you believe you can answer it correctly, output 'true'. If you are unsure whether you can answer it correctly, output 'false'. \
\nHere are some examples and the output format.[/INST]\n\n"

#     instruction = "[INST] You are a student being tested. For each given question, assess based on your knowledge whether you can answer it correctly.\
#  If you believe you can answer it correctly, output 'true'. If you are unsure whether you can answer it correctly and need to refer to external knowledge, output 'false'. \
# \nHere are some examples and the output format.[/INST]\n\n"

    # instruction = "[INST] You are a student being tested. For each given question, assess based on your knowledge whether you can answer it correctly. If you believe you can answer it correctly, output 'true'. If you are unsure or need to reference external knowledge to answer it, output 'false'.\nHere are some examples and the expected output format.[/INST]\n\n"


#     instruction = "[INST] You are a student being tested. For each given question, assess based on your knowledge whether you can answer it correctly.\
#  If you believe you can answer it correctly, output 'true'. If you are unsure whether you can answer it correctly or if the question is time-sensitive (i.e., the answer may vary over time), output 'false'. \
# \nHere are some examples and the output format.[/INST]\n\n"

#     instruction = "[INST] You are a student being tested. For each given question, assess based on your knowledge whether you can answer it correctly.\
#  If you believe you can answer it correctly, output 'true'. If you are unsure whether you can answer it correctly or if the answer of given question may changes over time, output 'false'. \
# \nHere are some examples and the output format.[/INST]\n\n"

#     instruction1="[INST] You are a student being tested. For each given question, assess based on your knowledge whether you can answer it correctly. If you believe you can answer it correctly, output 'true'. If you are unsure whether you can answer it correctly, output 'false'.\n\
# Here are some examples of questions you previously answered and whether your answers were correct."
     
#     demonsrate_instruct1 = ""

#     instruction2="\nBased on your previous performance, determine whether you can correctly answer the newly given questions below. [/INST]\n\n"

#     demonsrate_instruct2 = ""

#     for num,line in enumerate(reversed(demon_list)):
#         question = line["question"]
#         answer = line["not_ret_answer_result"]
#         one_demon_data = f"Question: {question}\nAnswer:{answer}\n"
#         if num<45:
#             demonsrate_instruct1+=one_demon_data
#         else:
#             demonsrate_instruct2+=one_demon_data

#     return instruction1+demonsrate_instruct1+instruction2+demonsrate_instruct2


    demonsrate_instruct = ""
   

    for line in reversed(demon_list):
        question = line["question"]
        answer = line["not_ret_answer_result"]
        one_demon_data = f"Question: {question}\nAnswer:{answer}\n"
        demonsrate_instruct+=one_demon_data

    return instruction+demonsrate_instruct


def test_instruct(icl,new_question):

    # test_question = f"\nHere is a new question.\nQuestion: {new_question}\nOnly output true or false.\n"

#     icl= f"[INST] You are a student being tested. For question:{new_question}, assess based on your knowledge whether you can answer it correctly.\
#  If you believe you can answer it correctly, output true. If you feel your answer might be incorrect, output false. Remember, only output true or false.[/INST]\n\n"
    # test_question = ""
    test_question = f"Question: {new_question}\nAnswer:"
    

    # test_question = f"\nHere is a new question.\nQuestion: {new_question}\nRemember, only output true or false. Do not output anything like 'Sure, I'm ready to answer.'\n"
    return icl+test_question





def self_judge(icl,data,model,tokenizer):
    for item in tqdm(data):

        input_ = test_instruct(icl,item["question"])
        # print(input_)
        out = single_inference_true_or_flase(input_,model,tokenizer,1)
        out = out.replace("\n"," ")



        if "true" in out:
            pred_true+=1
        elif "false" in out:
            pred_false+=1
        else:
            print("WRONG !!!!!!!!!!")


        if "true" in item["not_ret_answer_result"]:
            real_true +=1
            if "true" in out:
                true_acc+=1
        elif "false" in item["not_ret_answer_result"]:
            real_false+=1
            if "false" in out:
                false_acc+=1
        
    print("real_true:",real_true,end="    ")
    print("real_false:",real_false,end="    ")
    print("pred_true:",pred_true,end="    ")
    print("pred_false:",pred_false)

    print("false_acc: ",false_acc,end="    ")
    print("true_acc: ",true_acc,end="    ")

    print("all right:",false_acc+true_acc,end="    ")
    print("acc",(true_acc+false_acc)/(real_true+real_false))



if __name__ == '__main__':
     
    args = parse_args()
    DEV = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    model_name = args.model_dir
    print(model_name)
    print(args.test_file)
    print(DEV)

    dynamic = True

    # dataset select
    triviaqa = True
    taqa = False
    wq = False
    freshqa = False

    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,)

    # # true 1565 false 2089
    # print("****",tokenizer.encode(" false"))

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        # torch_dtype=torch.float16,
        load_in_4bit=False,
        trust_remote_code=True,

    )

    model = model.to(DEV)
    model.eval()

# test data load
    # with open("./model_without_ret_acc/llama2_7b_not_ret.json", 'r') as f:
    #     data = json.load(f)
    
#label load
    #triviaqa
    if triviaqa:
        # with open("/home/zhangyanan/zyn/lb/PreSelect/model_without_ret_acc/llama2_7b_ret_not_ret_demon_50_6w_train_data.json", 'r') as f:
        #     label_data = json.load(f)
            
        with open("/home/zhangyanan/zyn/lb/PreSelect/benchmarks/TriviaQA/llama2_7b_ret_not_ret_demon_100_aware_train.json", 'r') as f:
            label_data = json.load(f)
    #wq
    elif wq:
        with open("/home/zhangyanan/zyn/lb/PreSelect/benchmarks/wq/llama2_7b_ret_with_not_ret_result.json", 'r') as f:
            label_data = json.load(f)
    elif taqa:
        with open("/home/zhangyanan/zyn/lb/PreSelect/benchmarks/taqa/llama2_7b_ret_with_not_ret_result.json", 'r') as f:
            label_data = json.load(f)
    elif freshqa:
        with open("/home/zhangyanan/zyn/lb/PreSelect/benchmarks/freshqa/llama2_7b_ret_not_ret_demon_50_trivia.json", 'r') as f:
            label_data = json.load(f)


    if not dynamic:
        icl=construct_instruction("./model_without_ret_acc/demonstrate_10_1.json")
       
        # icl = only_instruction()
        print(icl)

        

    pred_true = 0 
    pred_false = 0
    real_true=0
    real_false = 0
    true_acc = 0
    false_acc = 0
    acc = 0
    response_acc = 0
    not_ret_acc = 0
    ret_acc = 0

    for item in tqdm(label_data):

        
        if dynamic:

            icl = construct_instruction_with_dynamic(item["demon_data"])

        input_ = test_instruct(icl,item["question"])

        # print(input_)
        out = single_inference_true_or_flase(input_,model,tokenizer,1,True)

#Subtract the original distribution of the model
        # wo_icl = only_instruction()
        # wo_context = test_instruct(wo_icl,item["question"])
        # input_with_context = input_
        # out = single_inference_true_or_flase_cad(wo_context,input_with_context,model,tokenizer)


        if "true" in out:
            pred_true+=1
            if item["not_ret_result"]=="true":
                response_acc += 1

        elif "false" in out:
            pred_false+=1
            if item["ret_result"]=="true":
                response_acc += 1
        else:
            print("WRONG !!!!!!!!!!")



        if "true" in item["not_ret_result"]:
            real_true +=1
            if "true" in out:
                true_acc+=1
        elif "false" in item["not_ret_result"]:
            real_false+=1
            if "false" in out:
                false_acc+=1
        else:
            raise KeyError
        

        if "true" in item["not_ret_result"]:
            not_ret_acc+=1
        
        if "true" in item["ret_result"]:
            ret_acc += 1


    print("real_true:",real_true,end="    ")
    print("real_false:",real_false,end="    ")
    print("pred_true:",pred_true,end="    ")
    print("pred_false:",pred_false)

    print("false_acc: ",false_acc,end="    ")
    print("false_acc %: ",false_acc/pred_false,end="    ")
    print("true_acc: ",true_acc,end="    ")

    print("all right:",false_acc+true_acc,end="    ")
    print("acc",(true_acc+false_acc)/(real_true+real_false))

    print("result_em_acc",response_acc/(real_true+real_false))
    print("not_ret_acc",not_ret_acc/(real_true+real_false))
    print("ret_acc",ret_acc/(real_true+real_false))
    
        
