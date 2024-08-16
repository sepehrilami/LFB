#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().run_line_magic('cd', 'LLaMA-Factory')


# In[14]:


import torch
import huggingface_hub
try:
  assert torch.cuda.is_available() is True
except AssertionError:
  print("Please set up a GPU before using LLaMA Factory: https://medium.com/mlearning-ai/training-yolov4-on-google-colab-316f8fff99c6")


# In[15]:


hf_auth = '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'
huggingface_hub.login(token = hf_auth)


# In[22]:


get_ipython().system('deepspeed --num_gpus 4 --master_port=9901 src/train_bash.py      --deepspeed "./examples/deepspeed/ds_z3_config.json"      --stage sft      --do_train      --model_name_or_path \'meta-llama/Llama-2-7b-chat-hf\'      --dataset factory_ready      --template default      --finetuning_type lora      --lora_target q_proj,v_proj      --output_dir "llama2_lora"      --overwrite_cache      --per_device_train_batch_size 4      --gradient_accumulation_steps 4      --lr_scheduler_type cosine      --logging_steps 10      --save_steps 1000      --learning_rate 5e-5      --num_train_epochs 3.0      --plot_loss      --fp16')


# In[26]:


get_ipython().system('cd /scratch/lora.n/STS_Project/')


# In[32]:


model = transformers.AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-2-7b-chat-hf',
        #"/scratch/lora.n/LLaMa2/config.py",
        #from_tf = True,
        cache_dir='/scratch/lora.n/LLaMa2',
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        token=hf_auth)


# In[ ]:




