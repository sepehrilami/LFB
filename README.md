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
mainloop.ipynb: This file replicates the original experiment and creates new datasets. It contains instructions for running either a fine-tuned or a pre-trained model. Our original fine-tuned model, l2tune, can be accessed by simply commenting and un-commenting the relevant lines. The finetuned small language model is provided in the /l2tune directory.

webul.ipynb: This file replicates the fine-tuning procedure we adopted to create l2tune. Note that an installation of LLaMafactory is necessary. The .json file factory_ready contains the original dataset used for fine-tuning, conveniently stored and organized according to LLaMafactory's syntax.

## Notebooks
The following notebooks were implemented for visualization and comparison of the three models performances across various scenarios.

in-sample-context-analysis.ipynb: Analysis of the results of in-sample game and contexts. The model names are 7b_org, 7b_finetuned, and 70b_reasoning. The 70b is an extra test (without reasoning) and is not included in the paper results. The figures are also provided in the /new_figs directory.

ood_game_analysis.ipynb: Similar to the main experiment, but with out-of-sample games (which were the same games but with different pay-off values. The pay-off matrix values were doubled). Results and figs are saved in /ood_game and /ood_game_figs directory.

oos_analysis: Similar to the main experiment, but with 3 out-of-sample contexts. Results and figs are saved in /oos_context and /oos_context_figs directory.

pgg_analysis: The analysis of out-of-sample game and context, which was a public pool game (PGG), called donation game. The results and figs are saved in the /pgg and /pgg_figs directory.
