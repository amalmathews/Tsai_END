# Capstone Project
## Objective and Guidelines
The capstone project is to write a transformer-based model that can write python code (with proper whitespace indentations). Here are the key points:

* The dataset can be found here. The dataset has 4600+ examples of English text to Python code.
* Write a transformer-based model that can learn to translate English to Python.
* There is no limit on the number of training epochs or total number of parameters in the model
* Train a separate embedding layer for python keywords that pays special attention to whitespaces, colon and other things (like comma etc)
* Your model has failed the capstone score if:
* your model fails to do proper indentation
* your model fails to use newline properly
* your model has failed to understand how to use colon (:)
* your model has failed to generate proper python code that can run on a Python interpreter and produce proper results
* You need to take care of some preprocessing things like:
* the dataset provided is divided into English and "python-code" pairs properly
* the dataset does not have anomalies w.r.t. indentations (like a mixed-use of tabs and spaces, or use of either 4 or 3 spaces, it should be 4 spaces only). Either use tabs only or 4 spaces only, not both
* the length of the "python-code" generated is not out of your model's capacity
* You need to submit a detailed README file that:
describes the data clean required
* your model architecture and salient features w.r.t. the model we wrote in the class
describes the loss function and how is it unique or improved than just using cross-entropy
* your data preparation strategy
* your data extension strategy (if you add more data)
* your "python-code" embedding strategy
* your evaluation metrics
* 25 example output from your model
* attention graph/images between text and "python-code"
* any additional point you'd want to add

## Capstone Solution

## Data Preparation
English_to python data is present in a single text file with Problem statement in English beginning as '#' and then the solution in python just after the problem statemnet.Each Problem statement in English is seperated by double "/n" newline character.
We need to separate these two as input and output and then tokenize both of them respectively.
  - When a line starts as a '#' consider that as input
  - All other lines to be considered as code.
  - replace '\n' as and ' ' as for easy tokenization of pyhton code and store as output.

We split the data as input and ouput and again cobonely split it as train,validation and test comma seperated files and save them.
We then define a tokeniser using python to tokenise them
We need to split the code wherever we find the following: split_points = ['<space>', ':', '<new_line>', '(', ')', '[', ']', '>', '<', '=', "'", '"', '.', '%', ',', '+', '-', '*', '\t', '{', '}', '/']


## Model Architecture and its salient features

#### Transformer Encoder Layer

  ![](https://raw.githubusercontent.com/bentrevett/pytorch-seq2seq/9479fcb532214ad26fd4bda9fcf081a05e1aaf4e/assets/transformer-encoder.png)

  The Encoders layers job is to map all input sequences into an abstract continuous representation that holds the learned information for that entire sequence. It contains 2 sub-modules, multi-headed attention, followed by a fully connected network. There are also residual connections around each of the two sublayers followed by a layer normalization.

##### Multihead Attention

 ![](https://raw.githubusercontent.com/bentrevett/pytorch-seq2seq/9479fcb532214ad26fd4bda9fcf081a05e1aaf4e/assets/transformer-attention.png)

 Multi-headed attention is a module in the transformer network that computes the attention weights for the input and produces an output vector with encoded information on how each word should attend to all other words in the sequence.The multi-headed attention output vector is added to the original positional input embedding. This is called a residual connection. The output of the residual connection goes through a layer normalization.
 The residual connections help the network train, by allowing gradients to flow through the networks directly. The layer normalizations are used to stabilize the network which results in substantially reducing the training time necessary. The pointwise feedforward layer is used to project the attention outputs potentially giving it a richer representation.

#### Transformer Decoder Layer

![](https://raw.githubusercontent.com/bentrevett/pytorch-seq2seq/9479fcb532214ad26fd4bda9fcf081a05e1aaf4e/assets/transformer-decoder.png)

The decoder’s job is to generate text sequences. The decoder has a similar sub-layer as the encoder. it has two multi-headed attention layers, a pointwise feed-forward layer, and residual connections, and layer normalization after each sub-layer.
These sub-layers behave similarly to the layers in the encoder but each multi-headed attention layer has a different job. The decoder is capped off with a linear layer that acts as a classifier, and a softmax to get the word probabilities.
The decoder then takes the output, add’s it to the list of decoder inputs, and continues decoding again until a token is predicted. For our case, the highest probability prediction is the final class which is assigned to the end token.
The decoder can also be stacked N layers high, each layer taking in inputs from the encoder and the layers before it. By stacking the layers, the model can learn to extract and focus on different combinations of attention from its attention heads, potentially boosting its predictive power.

## Output python code Embeddings

Once the data is splitted as mentioned in data preparation steps ,each of which can be used as a delimiter and is used of embeddings while training the model

## Loss Function

Default Cross entropy is only used


## Evaluation Metrics

For evaluating the model ***Bleu score*** has been used as metrics

BLEU is actually nothing more than a method to measure the similarity between two text strings. Human evaluations of machine translation outputs require considerable effort and are expensive. Human evaluations can take days or even weeks to finish so a new scoring system was developed to automate this process of evaluation. This method is commonly referred to as BLEU Score.

The BLEU metric scores a translation on a scale of 0 to 1. The metric attempts to measure adequacy and fluency in a similar way to a human would, e.g. does the output convey the same meaning as the input sentence, and is the output good and fluent target language? The closer to 1, the more overlap there is with a human reference translation and thus the better the system is. In a nutshell, the BLEU metric measures how many words overlap, giving higher scores to sequential words.

### Why is Bleau used for this project

BLEU allows  a way “to monitor the effect of daily changes to  systems in order to weed out bad ideas from good ideas.” When used to evaluate the relative merit of different system building strategies, BLEU can be quite effective as it provides very quick feedback and this enables MAchine Transltaion to quickly refine and improve English to python transaltions and continue to improve quality on a long term basis.

## Training and testing statistics

Total Epoch trained - 100

Epoch: 100 |
	Train Loss: 0.232 | Train PPL:   1.262
	 Val. Loss: 0.110 |  Val. PPL:   1.116

| Test Loss: 0.110 | Test PPL:   1.116 |

## 25 Examples

### Problem statement in English:
# write a function to calculate the current in the curcit where the resistance is r and voltage is v
### Ground Truth Python code:
def cal_current(resistance:float, voltage:float)->float:
    return voltage/resistance

### English to python prediction:
['def', '<space>', 'cal_current', '(', 'resistance', ':', 'float', ',', '<space>', 'voltage', ':', 'float', ')', '-', '>', 'float', ':', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'return', '<space>', 'voltage', '/', 'resistance', '<new_line>', '<eos>']
### Problem statement in English:
# write a python function to find the area of a circle , whose radius is given
### Ground Truth Python code:
def findarea(r):
    pi = 3.142
    return pi * (r*r)

### English to python prediction:
['def', '<space>', 'findarea', '(', 'r', ')', ':', '<space>', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'pi', '<space>', '=', '<space>', '3', '.', '142', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'return', '<space>', 'pi', '<space>', '*', '<space>', '(', 'r', '*', 'r', ')', '<new_line>', '<eos>']
### Problem statement in English:
# write a function to return the angualr veolcity based on augualr distance travelled in radian unit and time taken
### Ground Truth Python code:
def cal_angular_velocity(angular_dist:float,time:float)->float:
    return angular_dist/time

### English to python prediction:
['def', '<space>', 'cal_angular_velocity', '(', 'angular_dist', ':', 'float', ',', 'time', ':', 'float', ')', '-', '>', 'float', ':', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'return', '<space>', 'angular_dist', '/', 'time', '<new_line>', '<eos>']
### Problem statement in English:
# write a function that takes in height(m ) and weight(kg ) , calculates bmi and prints the comments
### Ground Truth Python code:
def bmi(height: 'meters', weight: 'kgs'):
    bmi = weight/(height**2)
    print('your bmi is: {0} and you are '.format(bmi), end='')
    if ( bmi < 16):
       print('severely underweight.')
    elif ( bmi >= 16 and bmi < 18.5):
       print('underweight.')
    elif ( bmi >= 18.5 and bmi < 25):
       print('healthy.')
    elif ( bmi >= 25 and bmi < 30):
       print('overweight.')
    elif ( bmi >=30):
       print('severely overweight.')

### English to python prediction:
['def', '<space>', 'time_elsaped', '(', ')', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'return', '<space>', 'inner', '(', ')', '<new_line>', '<eos>']
### Problem statement in English:
# write a program to swap first and last elements in a list
### Ground Truth Python code:
my_list = [1, 2, 3, 4, 5, 6]
my_list[0], my_list[-1] = my_list[-1], my_list[0]

### English to python prediction:
['my_list', '<space>', '=', '<space>', '[', '4', ',', '3', ',', '2', ',', '9', ',', '10', ',', '44', ',', '1', ']', '<new_line>', 'my_list', '.', 'sort', '(', 'reverse', '=', 'true', ')', '<new_line>', 'print', '(', 'f', "'", 'descending', '<space>', 'order', '<space>', 'list', ':', ',', '{', 'my_list', '}', "'", ')', '<new_line>', '<eos>']
### Problem statement in English:
# write a python function that takes in two numbers and returns their hcf
### Ground Truth Python code:
def hcf(num1, num2):
    smaller = num1 if num1 < num2 else num2
    for i in range(1, smaller+1):
        if (num1 % i == 0) and (num2 % i == 0):
            hcf = i
    return hcf

### English to python prediction:
['def', '<space>', 'hcf', '(', 'x', ',', '<space>', 'y', ')', ':', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'if', '<space>', 'x', '<space>', '>', '<space>', 'y', ':', '<new_line>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', 'smaller', '<space>', '=', '<space>', 'y', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'else', ':', '<new_line>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', 'smaller', '<space>', '=', '<space>', 'x', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'for', '<space>', 'i', '<space>', 'in', '<space>', 'range', '(', '1', ',', '<space>', 'smaller', '+', '1', ')', ':', '<new_line>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', 'if', '(', '(', 'x', '<space>', '%', '<space>', 'i', '<space>', '=', '=', '<space>', '0', ')', '<space>', 'and', '<space>', '(', 'y', '<space>', '%', '<space>', 'i', '<space>', '=', '=', '<space>', '0', ')', ')', ':', '<new_line>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', 'hcf', '<space>', '=', '<space>', 'i', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'return', '<space>', 'hcf', '<new_line>', 'num1', '<space>', '=', '<space>', '54', '<space>', '<new_line>', 'num2', '<space>', '=', '<space>', '24', '<new_line>', 'print', '(', "'", 'the', '<space>', 'l', '.', 'c', '.', '<space>', 'is', "'", ',', '<space>', 'num2', ',', '<space>', 'compute_hcf', '(', 'num1', ',', '<space>', 'num2', ')', ')', '<new_line>', '<eos>']
### Problem statement in English:
# write a function to calculate the total capacitance of capacitors in parallel in a given list
### Ground Truth Python code:
def cal_total_cap_in_parallel(cap_list:list)->float:
    return sum(cap_list)

### English to python prediction:
['def', '<space>', 'all_equal', '(', 'iterable', ')', ':', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'from', '<space>', 'itertools', '<space>', 'import', '<space>', 'groupby', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'return', '<space>', 'next', '(', 'g', ',', '<space>', 'none', ')', '<new_line>', '<eos>']
### Problem statement in English:
# write a function to calculate and return electricity bill . units used are given . price per unit is fixed and is increased after 750 units .
### Ground Truth Python code:
def calc_elect_bill(units):
    if units > 0:
        if units <= 750:
            return 5*units
        else:
            return 5*(750) + 7*(units-750)
    else:
        return -1

### English to python prediction:
['def', '<space>', 'time_elsaped', '(', ')', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'def', '<space>', 'inner', '(', ')', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'nonlocal', '<space>', '<new_line>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', 'return', '<space>', 'inner', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'inner', '<new_line>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', 'nonlocal', '<space>', 'perf_counter', '<new_line>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', 'return', '<space>', 'f', "'", 'outer', '<space>', 'between', '<space>', '{', 'fn', '.', '__name__', '}', '<space>', 'and', '<space>', '{', 'dt', '}', '<space>', 'inner', '(', ')', '<new_line>', '<space>', '<space>', '<space>', '<space>', '<space>', 'inner', '(', ')', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'print', '(', 'f', "'", 'outer', '<space>', 'function', '<space>', 'x', '<space>', 'is', '<space>', '{', 'fn', '.', '__name__', '}', "'", ')', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'inner', '(', ')', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'print', '(', 'f', "'", 'outer', '<space>', 'inner', ':', '{', 'fn', '(', ')', '}', '<space>', '{', 'outer', '(', ')', '<new_line>', '<space>', '<space>', 'inner', '(', ')', '<new_line>', '<eos>']
### Problem statement in English:
# write a python program to create multiple list
### Ground Truth Python code:
obj = {}
for i in range(1, 11):
    obj[str(i)] = []
print(f'create multiple list:{obj}')

### English to python prediction:
['my_list', '<space>', '=', '<space>', '[', '1', ',', '<space>', '2', ',', '<space>', '3', ',', '<space>', '4', ',', '<space>', '5', ',', '<space>', '6', ',', '<space>', '7', ',', '<space>', '8', ',', '<space>', '9', ',', '<space>', '10', ']', '<new_line>', 'print', '(', 'my_list', '[', ':', '5', ']', ')', '<new_line>', '<eos>']
### Problem statement in English:
# write a python function that returns the input list sorted in ascending order
### Ground Truth Python code:
def sort_ascending(list_to_be_sorted):
    return sorted(list_to_be_sorted)

### English to python prediction:
['def', '<space>', 'sort_ascending', '(', 'list_to_be_sorted', ')', ':', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'return', '<space>', 'sorted', '(', 'list_to_be_sorted', ')', '<new_line>', '<eos>']
### Problem statement in English:
# write a python function shifts and scales all numbers in the given list by the given mean and standard deviation
### Ground Truth Python code:
def shift_and_scale(list_of_nums, mean, std):
    return [ (x-mean) / std for x in list_of_nums ]

### English to python prediction:
['def', '<space>', 'get_weighted_average', '(', 'numbers', ',', '<space>', 'weightage', ')', ':', '<new_line>', '<space>', '<space>', '<space>', 'return', '<space>', 'sum', '(', 'x', '<space>', '*', '<space>', 'y', '<space>', 'for', '<space>', 'x', ',', '<space>', 'y', '<space>', 'in', '<space>', 'zip', '(', 'numbers', ',', '<space>', 'weightage', ')', ')', '<space>', '/', '<space>', 'sum', '(', 'weightage', ')', '<new_line>', '<eos>']
### Problem statement in English:
# write a python program to replace blank space to 1
### Ground Truth Python code:
my_string = 'balaji'
k = [print(i) for i in my_string if i not in 'aeiou']
print('not a vowel',k)

### English to python prediction:
['word', '<space>', '=', '<space>', "'", 'hello', '<space>', 'world', "'", '<new_line>', 'letter', '=', 'word', '[', '0', ']', '<new_line>', 'print', '(', 'f', "'", 'first', '<space>', 'charecter', '<space>', 'in', '<space>', 'string', ':', '{', 'letter', '}', "'", ')', '<new_line>', '<eos>']
### Problem statement in English:
# write a python program to print only digit or only apha charac in a given list
### Ground Truth Python code:
l=['good', 'oh!', 'excellent!', '#450']
print([n for n in l if n.isalpha() or n.isdigit()])

### English to python prediction:
['l', '<space>', '=', '<space>', '[', '1', ',', '<space>', '2', ',', '<space>', '3', ',', '<space>', '4', ',', '<space>', '5', ',', '<space>', '6', ',', '<space>', '7', ',', '<space>', '8', ',', '<space>', '9', ']', '<space>', '<new_line>', 'n', '<space>', '=', '<space>', '4', '<new_line>', 'x', '<space>', '=', '<space>', '[', 'l', '[', 'i', ':', 'i', '<space>', '+', '<space>', 'n', ']', '<space>', 'for', '<space>', 'i', '<space>', 'in', '<space>', 'range', '(', '0', ',', '<space>', 'len', '(', 'l', ')', ')', ']', '<space>', '<new_line>', 'print', '(', 'x', ')', '<space>', '<new_line>', '<eos>']
### Problem statement in English:
# write a python function to count how many times the predicate is true
### Ground Truth Python code:
def quantify(iterable, pred=bool):
    return sum(map(pred, iterable))

### English to python prediction:
['def', '<space>', 'countx', '(', 'lst', ',', '<space>', 'x', ')', ':', '<space>', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'return', '<space>', 'lst', '.', 'count', '(', 'x', ')', '<new_line>', '\t', '<new_line>', '\t', '<new_line>', '\t', '<new_line>', '<eos>']
### Problem statement in English:
# write a python function that returns the input list sorted in ascending order
### Ground Truth Python code:
def sort_ascending(list_to_be_sorted):
    return sorted(list_to_be_sorted)

### English to python prediction:
['def', '<space>', 'sort_ascending', '(', 'list_to_be_sorted', ')', ':', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'return', '<space>', 'sorted', '(', 'list_to_be_sorted', ')', '<new_line>', '<eos>']
### Problem statement in English:
# write a python function to return dictionary of two lists using zip
### Ground Truth Python code:
def dict_using_comp(list1, list2):
  dict_using_comp = {key:value for (key, value) in zip(list1, list2)}
  return dict_using_comp

### English to python prediction:
['def', '<space>', 'merge_lists', '(', 'l1', ':', 'list', ',', '<space>', 'l2', ':', 'list', ')', ':', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'return', '<space>', 'list', '(', 'zip', '(', 'l1', ',', 'l2', ')', ')', '<new_line>', '<eos>']
### Problem statement in English:
# write a python program to replace blank space to 1
### Ground Truth Python code:
print([i+j for i in 'abc' for j in 'def'])

### English to python prediction:
['word', '<space>', '=', '<space>', "'", 'hello', '<space>', 'world', "'", '<new_line>', 'letter', '=', 'word', '[', '0', ']', '<new_line>', 'print', '(', 'f', "'", 'first', '<space>', 'charecter', '<space>', 'in', '<space>', 'string', ':', '{', 'letter', '}', "'", ')', '<new_line>', '<eos>']
### Problem statement in English:
# write a python function to return dictionary of two lists using zip
### Ground Truth Python code:
def dict_using_comp(list1, list2):
  dict_using_comp = {key:value for (key, value) in zip(list1, list2)}
  return dict_using_comp

### English to python prediction:
['def', '<space>', 'merge_lists', '(', 'l1', ':', 'list', ',', '<space>', 'l2', ':', 'list', ')', ':', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'return', '<space>', 'list', '(', 'zip', '(', 'l1', ',', 'l2', ')', ')', '<new_line>', '<eos>']
### Problem statement in English:
# write a python program to check name exists in given list
### Ground Truth Python code:
names1 = ['amir', 'bala', 'chales']
for n in names1:
    name = n.lower()
    if 'amir' == name:
        print('yes name exists:',name)
    else:
        print('no')

### English to python prediction:
['my_list', '<space>', '=', '<space>', '[', '4', ',', '3', ',', '2', ',', '9', ',', '10', ',', '44', ',', '1', ']', '<new_line>', 'my_list', '.', 'sort', '(', 'reverse', '=', 'true', ')', '<new_line>', 'print', '(', 'f', "'", 'descending', '<space>', 'order', '<space>', 'list', ':', ',', '{', 'my_list', '}', "'", ')', '<new_line>', '<eos>']
### Problem statement in English:
# write a function to calculate the electrostatic force between two charged particles with charge q1 and q2 at a distance d apart
### Ground Truth Python code:
def cal_electrostatic_force(q1,q2,d):
    k = 9*(10**9)
    return (k*q1*q2)/(d**2)

### English to python prediction:
['def', '<space>', 'bit_div', '(', 'n', ',', '<space>', 'shift', ')', ':', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'return', '<space>', 'n', '<space>', '>', '>', '<space>', 'shift', '<new_line>', '<eos>']
### Problem statement in English:
# write a python program to explain the generator
### Ground Truth Python code:
def f11(x):
    yield x+1
g=f11(8)
print(next(g))

### English to python prediction:
['print', '(', "'", 'abcdefcdghcd', "'", '.', 'split', '(', "'", 'cd', "'", ',', '<space>', '2', ')', ')', '<new_line>', '<eos>']
### Problem statement in English:
# write a function to calculate compound interest , given p , r , t
### Ground Truth Python code:
def comp_int(p, r, t):
    amount = p * (1 + (r/100))**t
    interest = amount - p
    return interest

### English to python prediction:
['def', '<space>', 'comp_int', '(', 'p', ',', '<space>', 'r', ',', '<space>', 't', ')', ':', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'amount', '<space>', '=', '<space>', 'p', '<space>', '*', '<space>', '(', '1', '<space>', '+', '<space>', '(', 'r', '/', '100', ')', ')', '*', '*', 't', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'interest', '<space>', '=', '<space>', 'amount', '<space>', '-', '<space>', 'p', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'return', '<space>', 'interest', '<new_line>', '<eos>']
### Problem statement in English:
# python program to implement stooge sort
### Ground Truth Python code:
def stoogesort(arr, l, h):
    if l >= h:
        return
    if arr[l] > arr[h]:
        t = arr[l]
        arr[l] = arr[h]
        arr[h] = t
    if h - l + 1 > 2:
        t = (int)((h - l + 1) / 3)
        stoogesort(arr, l, (h - t))
        stoogesort(arr, l + t, (h))
        stoogesort(arr, l, (h - t))
arr = [2, 4, 5, 3, 1]
n = len(arr)
stoogesort(arr, 0, n - 1)
for i in range(0, n):
    print(arr[i], end= \' \')

### English to python prediction:
['def', '<space>', 'stoogesort', '(', 'arr', ',', '<space>', 'l', ',', '<space>', 'h', ')', ':', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'if', '<space>', 'l', '<space>', '>', '=', '<space>', 'h', ':', '<new_line>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', 'return', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'if', '<space>', 'arr', '[', 'l', ']', '<space>', '>', '<space>', 'arr', '[', 'h', ']', ':', '<new_line>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', 't', '<space>', '=', '<space>', 'arr', '[', 'l', ']', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'else', ':', '<new_line>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', 't', '<space>', '=', '<space>', 'arr', '[', 'l', ']', '<new_line>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', 'for', '<space>', 'i', '<space>', 'in', '<space>', 'range', '(', '0', ',', '<space>', 'len', '(', 'arr', ')', ')', ':', '<new_line>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', 'if', '<space>', 'arr', '[', 'i', ']', '<space>', '>', '<space>', 'max', ':', '<new_line>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', 't', '<space>', '=', '<space>', 'arr', '[', 'i', ']', ',', '<space>', 'arr', '[', 'i', ']', '<new_line>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', 'break', '<new_line>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', 'if', '<space>', 'arr', '[', 'i', ']', '<space>', '>', '<space>', 'arr', '[', 'l', ']', ':', '<new_line>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', '<space>', 'arr', '[', 'l', ']', ',', '<space>', 'arr', '[', 'l', ']', '<space>', '+', '<space>', '=', '<space>', '1', '<new_line>', '<space>', '<space>', '<space>', '<space>', 'return', '<space>', 'arr', '<new_line>', 'arr', '<space>', '[', '34', ',', '<space>', '[', '2', ',', '<space>', '3', ',', '<space>', '5', ',', '<space>', '1', ']', '<new_line>', 'print', '(', "'", 'sorted', '<space>', 'array', '<space>', 'is', '<space>', 'not', '<space>', 'a', '<space>', ':', '<space>', "'", ',', '<space>', 'str', '(', 'arr', ')', ')', '<new_line>', '<eos>']
### Problem statement in English:
# write a python program to sort tuple values
### Ground Truth Python code:
a=(2,3,1,5)
tuple_sorted = sorted(a)
print(tuple(tuple_sorted))

### English to python prediction:
['print', '(', "'", 'the', '<space>', 'original', '<space>', 'dictionary', '<space>', 'is', '<space>', ':', '<space>', "'", '<space>', '+', '<space>', 'str', '(', 'test_dict', ')', ')', '<space>', '<new_line>', '<space>', '<space>', '<new_line>', '<eos>']
### Problem statement in English:
# write a python function that would filter a list of dictionaries where a specified key equals given value , list_of_dictionaries , key and value are inputs to this function .
### Ground Truth Python code:
def filter_with_key_value(list_of_dicts, key, value):
    return list( filter( lambda x: x.get(key) == value, list_of_dicts ) )

### English to python prediction:
['def', '<space>', 'join_string_parts', '(', 'str_list', ')', ':', '<new_line>', '<space>', '<space>', '<space>', 'return', '<space>', "'", '<space>', "'", '.', 'join', '(', 'str_list', ')', '<new_line>', '<eos>']
