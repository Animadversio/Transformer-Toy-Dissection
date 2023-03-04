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

* Will it help if we allow it to output the answer in the reverse digit order => it will have access to lower digit when outputing higher digits?
* Further will it help to have intermediate steps as scratch pad?


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

At epoch 134, the model has not learned multiplication of even single digit. 
```
1 * 3 = 3, model ans [3, 4, 2, 33, 4] accuracy: 0.20
2 * 3 = 6, model ans [420, 1, 4, 7417, 7] accuracy: 0.00
10 * 20 = 200, model ans [200, 200, 200, 200, 200] accuracy: 1.00
10 + 20 = 30, model ans [30, 30, 30, 30, 30] accuracy: 1.00
10 - 20 = -10, model ans [90, -10, -10, 10, 70] accuracy: 0.40
5 * 155 = 775, model ans [555, 675, 785, 565, 775] accuracy: 0.20
3 * 30 = 90, model ans [33, 35, 34, 36, 90] accuracy: 0.20
5 * 12 = 60, model ans [76, 58, 58, 70, 70] accuracy: 0.00
9 * 11 = 99, model ans [179, 171, 89, 119, 199] accuracy: 0.00
19 + 13 = 32, model ans [32, 42, 42, 42, 32] accuracy: 0.40
11 + 111 = 122, model ans [122, 122, 122, 122, 122] accuracy: 1.00
11 * 112 = 1232, model ans [1232, 1232, 1232, 1232, 1232] accuracy: 1.00
```

At epoch 160
```
1 * 3 = 3, model ans [3, 3, 8, 5, 9] accuracy: 0.40
2 * 3 = 6, model ans [4, 6, 6, 5, 6] accuracy: 0.60
10 * 20 = 200, model ans [200, 200, 200, 200, 200] accuracy: 1.00
10 + 20 = 30, model ans [20, 30, 30, 30, 30] accuracy: 0.80
10 - 20 = -10, model ans [70, 0, -9, 0, 0] accuracy: 0.00
5 * 155 = 775, model ans [875, 675, 775, 875, 875] accuracy: 0.20
3 * 30 = 90, model ans [90, 90, 86, 90, 144] accuracy: 0.60
5 * 12 = 60, model ans [56, 51, 52, 65, 51] accuracy: 0.00
9 * 11 = 99, model ans [199, 99, 99, 99, 99] accuracy: 0.80
19 + 13 = 32, model ans [32, 32, 30, 32, 32] accuracy: 0.80
11 + 111 = 122, model ans [122, 122, 182, 122, 122] accuracy: 0.80
11 * 112 = 1232, model ans [1232, 1232, 1232, 1232, 1232] accuracy: 1.00
916 + 316 = 1232, model ans [1232, 1232, 1232, 1232, 1232] accuracy: 1.00
782 + 520 = 1302, model ans [1302, 1302, 1302, 1302, 1302] accuracy: 1.00
437 * 335 = 146395, model ans [145695, 147595, 144795, 144795, 148395] accuracy: 0.00
683 * 66 = 45078, model ans [45078, 45278, 45178, 45158, 45078] accuracy: 0.40
219 - 228 = -9, model ans [-9, -9, -9, -9, -99] accuracy: 0.80
243 + 350 = 593, model ans [593, 593, 593, 593, 593] accuracy: 1.00
500 - 56 = 444, model ans [444, 444, 444, 444, 444] accuracy: 1.00
318 + 933 = 1251, model ans [1251, 1251, 1251, 1251, 1251] accuracy: 1.00
327 + 606 = 933, model ans [933, 933, 933, 933, 933] accuracy: 1.00
859 * 282 = 242238, model ans [243238, 240238, 242838, 244038, 242038] accuracy: 0.00
720 + 88 = 808, model ans [808, 808, 808, 808, 808] accuracy: 1.00
151 + 7 = 158, model ans [158, 158, 158, 158, 158] accuracy: 1.00
371 * 198 = 73458, model ans [72858, 73458, 71658, 71458, 73658] accuracy: 0.20
283 * 416 = 117728, model ans [119428, 119428, 119528, 117228, 118828] accuracy: 0.00
582 - 375 = 207, model ans [207, 207, 207, 207, 207] accuracy: 1.00
986 + 642 = 1628, model ans [1628, 1628, 1628, 1628, 1628] accuracy: 1.00
105 + 663 = 768, model ans [768, 768, 768, 768, 768] accuracy: 1.00
726 * 367 = 266442, model ans [261042, 264142, 260242, 268442, 262042] accuracy: 0.00
819 - 110 = 709, model ans [709, 709, 709, 709, 709] accuracy: 1.00
721 * 978 = 705138, model ans [706638, 703438, 704738, 704838, 704338] accuracy: 0.00
329 + 101 = 430, model ans [430, 430, 430, 430, 430] accuracy: 1.00
738 + 316 = 1054, model ans [1054, 1054, 1054, 954, 1054] accuracy: 0.80
```
### Scratch pad reasoning (Intermediate step supervision)



Related works
---
[CSAIL blog: Notes on Teaching GPT-3 Adding Numbers](https://lingo.csail.mit.edu/blog/arithmetic_gpt3/)
