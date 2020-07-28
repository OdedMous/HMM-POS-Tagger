# HMM POS Tagger

## **Goal**
This program implements and compares several versions of
an HMM POS (Part of Speech) tagger:

1) MLE (as a baseline model)
2) bigram HMM
3) bigram HMM + add-1 smoothing
4) bigram HMM + using pseudowords
5) bigram HMM + add-1 smoothing + using pseudowords

**Evaluation - comparing the above models is based on calculating:**
- Error rate of known words
- Error rate of unknown words
- Total error rate

**Dataset for train and test sets:**
Brown corpus - news category

## **Structure and Usage**
**code**: <br/>
*main.py* - extracts the data and runs the program. <br/>
*models.py* - implementation of baseline and HMM models.  <br/>
*pseudowords.py* - contains function that maps words to their pseudowords. <br/>
*word_counting.py* - contains data structures that stores word and tag counts for the text. 

**Usage:**
python main.py

**Output:**
An Excel file is saved in the current directory.
It contains the errors results and a chart which visualizes it.
   
**Example results:** <br/>
![results example](../master/images/res.png)

 Note that the results might change a little in each run since the program use viterbi algorithm which
assign random tags for unkown words.

-----------------------------------------------------------------------
## **Why POS tagging is an important task in NLP?**
Part of speech tag of a word can expose the context in which the word is used.
Therefore POS tagging is often a first step in solving other tasks in NLP, such as Semantic Role Labeling and Machine Translation.

## **Key concepts and terms used in this program**
**Baseline model:**
A very simple modle which designed to provide a simple solution for the task. If our next (and more complicated) models will not give us better results than this baseline -  we should reconsider using them (or check if we have implementation mistakes). In this program we took as a baseline the MLE (most likelihood estimator) model, which simply assign each word the tag that apperas with it the most in the training set.

**HMM tagger:** 
Hidden Markov Model is a probabilistic sequence classifier, and it's also a generative model.
In the context of POS tagging, this model makes two assumptions: 
1. The probability to genertate a word depends only in the current chosen tag. This probability is called "emmision probability".
2. The probability to genertate the next tag depends only in the n last chosen tags (this assumption is    called "Markov assumption", and in this program we chose n=2, aka bigram HMM). This probability is    called "transition probability" <br />                  
More details on HMM can be found in the excellent book "Speech and Language Processing", by Dan Jurafsky and James H. Martin.

**Unkown words:**
Words that doesn't appear in the training set but does appear in the test set. Hence it's harder to  predict thier tag since we relay heavly on the apperance of the words in the training set.

**Pseudowords:**
A method for dealing with unknown words. The idea is to convert words from both training and tests sets into speficic predifined words. Hopefully, after the conversion, many unknown words will appear also in the training set.

**Add-1 smoothing:**
Since our models depends on counting apperance of pairs of words (because we use bigram HMM), in the training set, if some pair from test set won't appear there, then we assign probability 0 to this "unseen event", and hence assign probability 0 to the current sequence which we try to assign to it a probability. Furthermore, pairs that does appear in training set will get higher probability - and hence there's a risk to overfitting. Add-1 smoothing is a technique for dealing these problems by assign each count the value 1. Note that this is not the preferred solution for these problems (see other smoothing methods such as Kneser-Ney Smoothing)

**The Viterbi algorithm**
A dynamic programming algorithm for finding the most likely sequence of hidden states (=tags in our case). Its advantage is its run time: if the number of optional tags is K and the length of the words sequence is N, than the **inference** (prediction) phase (calculating the probability for each sequance of tags, and return the most probability one) of a **brute-force method** takes O(K^N). However, viterbi algorithem takes O(N * K^m) for a m-gram HMM model.

## **Some implementation details**

**Object "Counts" from word_counting.py:**
It contains data structures that stores important information about the text such as word and tag counts for the text. 
Two advantages obtained by that:
1. Run time - in a one-time single pass on the data we extract usful information that makes calculations run faster later in the program. 
2. Readability - we can access those structures from anywhere in the program, instead of reconstruct them over and over agian. 
Disadvantages:
1. Memorey

* **Pseudo-words**: 
We map all low-frequency words from training set, and all unknown words (from test set) to pseudo-words.
My main ideas was to map a word by:
- its suffix/prefix.
- if it contains a number
- some more little spelling things, such as if the word has capital first letter or if the word contains $.
Furtter thinking one should check:
- combaine suffix/prefix of a word with its capital first letter.
- consider more carefully special symbols like '.' and '-'.
- consider more carfully numbers ("4th" and "forth" may get the same pseudo-word, but "4-years" not).
- and more...
