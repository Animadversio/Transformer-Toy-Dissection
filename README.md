# Transformer-Toy-Dissection

## Problem set up
In this project, we treat arithmetics as a sequence learning problem. 
```
12 + 56 = 68
```
where we tokenize individual digit `0-9` and operators `+-*/= ` and transform the string as a sequence of tokens. 

Using this formulation, we train transformers to predict and complete these sequences. Surely, as a expressive neural network, it should be easiy to fit the distributions. But will it generalize to unseen sequences? will it generalize sequences of longer length? 
If so, how does transformers solve arithmetic problems? is it the same solution as the human being?


Related works
---
[CSAIL blog: Notes on Teaching GPT-3 Adding Numbers](https://lingo.csail.mit.edu/blog/arithmetic_gpt3/)
