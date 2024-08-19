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
The data for creating the prompts are available in the /data directory.

## Rerun the experiment
This experiment was done using Groq platform for loading the LLMs. However, with the introduction of LlaMa3 models, LLaMa2 models are no longer available through Groq APIs. Therefore, to rerun the experiments, we need to use a system which had sufficient GPU power.

## Description of each file
main_experiment.ipynb: In this file create the prompts based on games and contexts, and prompt the LLMs and collect their results.

webul.ipynb: H

