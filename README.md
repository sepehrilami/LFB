# Large Model Strategic Thinking, Small Model Efficiency: Transferring Theory of Mind in Large Language Models

# Summary
In this study we finetune small language models (SLMs) to improve their decision making and strategic choice actions by learning from a large language model (LLM) and enhance its performance in a specific domain, without the necessity of increasing the model size, number of parameters, etc.

# Keywords
LLM, Game Theory, Theory of mind, Fine-tuning, Strategic Choice, Decision-making

# Code Details
## Requirements
To install the requirements, run the following command:
```pip install -r requirements.txt```

## Data and Prompts
All contextual and game prompts are available in the /data directory. The /results directory contains all our results from within-sample experiments, while /oos_context and /ood_game contain results from out of sample contexts and games, respectively. Finally, /pgg contains data for the public good game experiment.  

## Rerun the experiment
This experiment was done using the Groq platform for loading the LLMs. However, with the introduction of LlaMa3 models, LLaMa2 models are no longer available through Groq APIs. Therefore, to rerun the experiments, a system with sufficient GPU is required, either locally or via hosting services. In our code, we access LLaMa2-7b and 70b via Huggingface. Local or remote fine-tuned model can also be accessed via our code.  

## Replication details
mainloop.ipynb: This file replicates the original experiment and creates new datasets. It contains instructions for running either a fine-tuned or a pre-trained model. Our original fine-tuned model, l2tune, can be accessed by simply commenting and un-commenting the relevant lines. 

webul.ipynb: This file replicates the fine-tuning procedure we adopted to create l2tune. Note that an installation of LLaMafactory is necessary. The .json file factory_ready contains the original dataset used for fine-tuning, conveniently stored and organized according to LLaMafactory's syntax.

