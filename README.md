# Can toy Transformers learn to do arithmetic?

## Problem set up
In this project, we treat arithmetics as a sequence learning problem. 
```
12 + 56 = 68
```
where we tokenize individual digit `0-9` and operators `+-*/= ` and transform the string as a sequence of tokens. 

Using this formulation, we train transformers to predict and complete these sequences. Surely, as an expressive neural network, it should be easiy to fit the distributions of sequences. But,  
* Will it generalize to unseen sequences? 
* Will it generalize to sequences of longer length? 
* If so, how does transformers solve arithmetic problems? is it the same solution as the human being? (network dissection & mechanistic interpretability)

Even if the fitting is perfect, there are some interesting things to note during its learning. During the learning process
* Which operation is learnt first? (I hypothesize addition and subtraction operation is easier to learn than the multiplication, while division is the hardest)
* From the error patterns, what aspect of arithmetic is learnt first?
  * does it discover tricks like 5 times anything ends in 5. 
  * does it discover 1 times anythin ends in anything?
  * does it learn to figure out the sign of answer early on? 
  * does it converge to the correct answer first in scale (number of digits) or in values or suddenly jumps to the correct answer?
* If it makes error does it error like human? e.g. recursive carry overs? 

Beyond behavioral level, internally, we will also investigate the following questions
* Does network configure itself into different modes with different operators? (+ - can be really different from /)
* i.e. different representation of digits or different operations on digits?

Further,
* Does the network's solution depend on its depth or width?

If it does not learn well, will curriculum learning help?
* i.e. if the model learns single digit add, and multiplication first, will it learn faster on more digits multiplication?
* will it benefits from learning **the order of digits, by sorting them**? 

## Results
### Sequence modelling training won't generalize
If the network is trained only on 2 digits addition / subtraction, it won't generalize to 3 digits. 

Similarly if the network is trained on 3 digits add / subtract, it won't generalize beyond that. 

### Sequence of learning: multiplication is harder
It seems that the network learns the addition and subtraction first. At epoch 61
```
10 * 20 = 200, model ans [200, 219, 124, 100, 85] accuracy: 0.20
10 + 20 = 30, model ans [43, 35, 38, 42, 44] accuracy: 0.00
10 - 20 = -10, model ans [-10, -16, -22, -1, -19] accuracy: 0.20
5 * 155 = 775, model ans [1177, 7046, 5178, 531, 5346] accuracy: 0.00
3 * 30 = 90, model ans [22, 114, 21210, 113, 20] accuracy: 0.00
5 * 12 = 60, model ans [16, 277, 30, 22, 11231] accuracy: 0.00
9 * 11 = 99, model ans [47, 13, 27, 197, 90] accuracy: 0.00
19 + 13 = 32, model ans [28, 40, 121, 48, 28] accuracy: 0.00
11 + 111 = 122, model ans [124, 114, 113, 121, 122] accuracy: 0.20
11 * 112 = 1232, model ans [1390, 1342, 1229, 1542, 1020] accuracy: 0.00
765 + 946 = 1711, model ans [1711, 1711, 1711, 1711, 1711] accuracy: 1.00
544 - 858 = -314, model ans [-314, -324, -314, -314, -314] accuracy: 0.80
995 + 918 = 1913, model ans [1913, 1913, 1913, 1913, 913] accuracy: 0.80
```

For example at epoch 68, the model can do 3 digits addition and subtraction with high accuracy, while multiplication is nearly 0.
```
177 + 696 = 873, model ans [873, 873, 873, 873, 873] accuracy: 1.00
742 + 672 = 1414, model ans [1414, 1414, 1414, 1414, 1414] accuracy: 1.00
391 + 933 = 1324, model ans [1324, 1324, 1324, 1324, 1324] accuracy: 1.00
822 + 466 = 1288, model ans [1288, 1288, 1288, 1288, 1288] accuracy: 1.00
789 + 968 = 1757, model ans [1757, 1757, 1757, 1757, 1757] accuracy: 1.00
687 - 324 = 363, model ans [363, 363, 363, 363, 363] accuracy: 1.00
183 + 403 = 586, model ans [586, 586, 586, 586, 586] accuracy: 1.00
280 - 321 = -41, model ans [-41, -41, -41, -41, -41] accuracy: 1.00
231 - 180 = 51, model ans [51, 52, 51, 51, 51] accuracy: 0.80
935 - 89 = 846, model ans [846, 836, 846, 846, 846] accuracy: 0.80
11 * 112 = 1232, model ans [1432, 1221, 1842, 1632, 1132] accuracy: 0.00
932 * 742 = 691544, model ans [687464, 687524, 687664, 686164, 683804] accuracy: 0.00
412 * 745 = 306940, model ans [307000, 301560, 313940, 316490, 300340] accuracy: 0.00
```
Note that 
* even when the multiplication answer is wrong, the model learned the statistical association s.t. 
  * it gets the number of digits right; 
  * the first and last digit is correct. 
  * while the other digits suffers from lots of carry over errors. 

### Scratch pad reasoning (Intermediate step supervision)



Related works
---
[CSAIL blog: Notes on Teaching GPT-3 Adding Numbers](https://lingo.csail.mit.edu/blog/arithmetic_gpt3/)
