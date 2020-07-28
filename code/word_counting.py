import numpy as np
from collections import defaultdict

"""
in this program the data (sentences) is represented by tuples of the form (word, tag)
"""
WORD = 0
TAG = 1


class Counts:
    """
    This class contains data structures that stores word and tag counts
    for the given texts.
    """

    def __init__(self, training, test):
        """
        Initialize data structures.
        :param training: current training set.
        :param test: current test set.
        """

        """set that contains all distinct words from training set"""
        self.known_words = set()

        """set that contains all distinct words from test set which doesn't appear in the training set"""
        self.unknown_words = set()

        """numpy array of shape(N,) which contains all distinct words from training set"""
        self.order_known_words = np.array([])

        """set that contains all distinct tags from training set"""
        self.training_tags = set()

        """numpy array of shape(N,) which contains all distinct tags from training set"""
        self.order_training_tags = np.array([])

        """numpy array of shape(N,) which contains all distinct tags from test set"""
        self.order_test_tags = np.array([])

        """dict which counts num of occurrence for each word in training set"""
        self.count_training_words = defaultdict(int)

        """dict which counts occurrence of tags in the training set."""
        self.count_training_tags = defaultdict(int)

        """dict which counts occurrence of all pairs (word, tag) in the training set."""
        self.count_word_tag_pairs = defaultdict(int)

        temp = []
        temp2 = []

        for word in training: # note that word is actually a tuple of (word, POS)
                if word[WORD] not in self.known_words:
                    temp2.append(word[WORD])

                if word[TAG] not in self.training_tags:
                    temp.append(word[TAG])

                self.known_words.add(word[WORD])
                self.training_tags.add(word[TAG])
                self.count_training_words[word[WORD]] += 1
                self.count_training_tags[word[TAG]] += 1
                self.count_word_tag_pairs[word] += 1

        self.order_known_words = np.array(temp2)
        self.order_training_tags = np.array(temp)

        temp3 = []
        for word in test:
            if word[WORD] not in self.known_words:
                self.unknown_words.add(word[WORD])

            if word[TAG] not in temp3:
                temp3.append(word[TAG])

        self.order_test_tags = np.array(temp3)
