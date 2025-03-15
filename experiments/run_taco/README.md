# TACO Experiments

The goal here is try to explore datamodels as a retriever for context-mixing scenarios in coding generation  
The choosen benchmark for this scenairo was [TACO]{https://github.com/FlagOpen/TACO}

## Experiments

Some different thins will be explored here, this benchmark will be explored in the following directions:
- Baseline Results Distribution: It will be sampled 20 examples from each difficulty, then we will run the pass@k evalutions for 200 different generations of each one of them
- Best Context Scenario: Cheery-picked example where it's evaluated the code generation passing a designed context for that problem. One try will be with examples form the dataset and other with coding examples generated using LLM
