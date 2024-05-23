#!/usr/bin/env python
# coding: utf-8



from torch import cuda, bfloat16
import pickle
import transformers
from langchain import PromptTemplate,  LLMChain
import time

#groq_auth = "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"
model_id = 'meta-llama/Llama-2-7b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16,
    load_in_8bit_fp32_cpu_offload=True,
)

# begin initializing HF items, need auth token for these
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    token=hf_auth
)

try:

    model = transformers.AutoModelForCausalLM.from_pretrained(
        #model_id,
        './l2tune',
        #"/scratch/lora.n/LLaMa2/config.py",
        #from_tf = True,
        cache_dir='/scratch/lora.n/STS_Project',
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        token=hf_auth
    )


except Exception as e:
    print("ERROR:", e)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    token=hf_auth
)
tokenizer.pad_token_id = tokenizer.eos_token_id


stop_list = ['\nHuman:', '\n```\n']

stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids

import torch

stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
stop_token_ids

from transformers import StoppingCriteria, StoppingCriteriaList

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])
stopping_criteria

from langchain.llms import HuggingFacePipeline

pipe = transformers.pipeline(model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, return_full_text=True,  
task='text-generation', stopping_criteria=stopping_criteria,  temperature=0.8, max_new_tokens=600, repetition_penalty=1.1)

llm = HuggingFacePipeline(pipeline=pipe)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<>\n", "\n<>\n\n"

def get_prompt(instruction, new_system_prompt):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def GameRepeat(context, code, floor, loadflag = False):
    #initialize game and scenario
    scen = open(context +'.txt').read()
    game = open(code  +'.txt').read()
    
    with open('7b-fined-results-oos.txt', 'rb') as file:
        try:
            loc_res = pickle.load(file) 
        except EOFError:
            loc_res = {}
            print("File empty!")
        
    sys_msg = B_SYS + "Consider the proposed scenario and act as if you were taking part in it. \n"+ scen + E_SYS
    #initalize user message
    cmds = B_INST + " Respond to the following using exactly one letter to denote your choice. Your answer must either consist of the letter 'C' for strategy C or 'D' for strategy D." + E_INST
    human_msg = cmds + "\nUser: {user_input}"
    template = get_prompt(human_msg,  sys_msg)
    prompt = PromptTemplate(input_variables=["user_input"], template=template)

    lm = LLMChain(llm=llm,prompt=prompt,verbose=False)
    
    stratlist = []
    if not context+"_"+code in loc_res:
        loc_res.update({context+"_"+code:stratlist})
    for i in range(floor,300):
        rz = lm.predict(user_input = game)
        stratlist.append(rz)
        loc_res[context+"_"+code].append(rz)
        time.sleep(3)
        with open('7b-fined-results-oos.txt', 'wb') as file:
            pickle.dump(loc_res, file)
            
    return stratlist


scenarios = ['sports', 'roomsharing', 'ventcap'] 
games = ['prison', 'delight', 'staghunt', 'snowdrift']
results_7b = {}

try:
   with open('7b-fined-results-oos.txt', 'rb') as file:
       results_7b = pickle.load(file)
except Exception as e:
    pass
       
for scenario in scenarios:
    for game in games:
        lowl = 0
        if scenario+"_"+game in results_7b:
            if len(results_7b[scenario+"_"+game]) < 300:
                lowl = len(results_7b[scenario+"_"+game])
            else:
                continue
        gamma = GameRepeat(scenario, game, lowl)
        results_7b.update({scenario+"_"+game:gamma})
        with open('7b-fined-results-oos.txt', 'wb') as file:
            pickle.dump(results_7b, file)





