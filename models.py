from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import random
from word_counting import *

class Model(ABC):
    """
    This abstract class defines a general scheme for sequence tagging models.

        Attributes:
        counts     An object which contains data structures that stores word and tag counts
                   for some text.

    """

    def __init__(self, counts):
        self.counts = counts

    def calculate_error_rate(self, predicted_tags, test):
        """
        This function computes the error rate for a given sequence of tags
        (which computed by some algorithm like MLE_baseline or viterbi in order
        to predict tags for the test set).
        The error is calculate for known words, for unknown words and for the
        total words in the *test set*.
        Formula: error rate = 1 - accuracy
                 accuracy = #correct_tags / #word_tokens_in_test_set
        :param predicted_tags: a list of tags.
               predicted_tags[i] is the predicted tag for the i'th word in the test set.
        :param test - current test set
        :return known_err, unknown_err, total_err
        """

        known_correct_tags, unknown_correct_tags = 0, 0

        num_word_tokens_test_set = len(test)

        for i, word_tuple in enumerate(test):
            if word_tuple[TAG] == predicted_tags[i]:
                if word_tuple[WORD] in self.counts.known_words:
                    known_correct_tags += 1
                else:
                    unknown_correct_tags += 1

        known_accuracy = known_correct_tags / num_word_tokens_test_set
        unknown_accuracy = unknown_correct_tags / num_word_tokens_test_set
        total_accuracy = (known_correct_tags + unknown_correct_tags) / num_word_tokens_test_set

        return [1 - known_accuracy, 1 - unknown_accuracy, 1 - total_accuracy]

    @abstractmethod
    def errors(self, *args):
        """
        Calculates the errors of the model.
        :return known_err, unknown_err, total_err
        """
        pass


class Baseline_Model(Model):
    """
    Implementation of a baseline model for sequence labeling task.
    The chosen model is MLE.
    """

    def MLE_tagger(self, test, text):
        """
        Using the training set, this function computes for each word the tag that
        maximizes p(tag|word), based on maximum likelihood estimation.
        Meaning it assigning each token to the tag that occurred with it most often in the training set.
        This function assumes that the most likely tag of all the unknown words is 'NN'.
        Formula: for each word, P(tag|word) = max_tag[Count(word, tag)/Count(word)]
                 *Actually there is no need to consider Count(word) in the calculation,
                  as it equal for each tag. That is, we just want to find max_tag, and not the probability itself.
        :return: predicted_tags - a list of tags.
                 predicted_tags[i] is the predicted tag for the i'th word in the test set.
                 *Note that by definition of MLE baseline, each token with the same word type
                  will receive the same tag.
        """

        predicted_tags = {}
        original_vocabulary = {word[WORD] for sent in text for word in sent}

        for word in original_vocabulary:
            if word not in self.counts.known_words:
                # we assume that the most likely tag of all the unknown words is “NN”.
                predicted_tags[word] = 'NN'
                continue
            # dict which contains counts of pairs of the form ('word', tag) from training set
            # (all pairs which the 'word' is the same for them)
            temp = {key: self.counts.count_word_tag_pairs[key] for key in self.counts.count_word_tag_pairs.keys() if key[WORD] == word}

            # get the tag that occur with 'word' the most in the training set
            predicted_tags[word] = (max(temp))[TAG]

        # convert test_set from sentences of tuples to single sequence of words
        test_tokens = [item[0] for item in test]

        final = [predicted_tags[word] for word in test_tokens]
        return final

    def errors(self, test, text):
        """
        Calculates the error rate for MLE tag baseline.
        :return known_err, unknown_err, total_err
        """
        tags_results = self.MLE_tagger(test, text)
        known_err, unknown_err, total_err = self.calculate_error_rate(tags_results, test)
        return known_err, unknown_err, total_err


class HMM_Model(Model):
    """
    Implementation of a HMM model for sequence labeling task.
    """

    def count_consecutive_tags(self, training):
        """
        Return a dict which contains:
        result[(tag_n, tag_m)] = Count(tag_n, tag_m)
        """
        result = defaultdict(int)

        for i in range(len(training)-1):
            result[(training[i][TAG], training[i+1][TAG])] += 1

        return result

    def transition_probabilities(self, training):
        """
        Computes the transition probabilities of a bigram HMM tagger
        on the training set, using MLE.
        :return: t - a 2D numpy array of shape (#training_tags, #training_tags)
         which contains the probabilities to get tag_i given tag_j was previous:
         t[i][j] =P(tag_i|tag_j) = Count(tag_j, tag_i)/Count(tag_j)
         where:
         i = np.where(order_training_tags == tag_i)[0][0]
         j = np.where(order_training_tags == tag_j)[0][0]
        """

        count_tags_pairs = self.count_consecutive_tags(training)

        #number of distinct tags in training set
        num_tags = self.counts.order_training_tags.shape[0]
        t = np.empty((num_tags, num_tags), dtype=float)

        for i in range(num_tags):
            tag1 = self.counts.order_training_tags[i]
            for j in range(num_tags):
                tag2 = self.counts.order_training_tags[j]
                t[i, j] = count_tags_pairs[(tag1, tag2)] / self.counts.count_training_tags[tag2]

        return t

    def emission_probabilities(self, add_one=False):
        """
        Computes the emission probabilities of a bigram HMM tagger
        on the training set, using MLE.
        :param add_one: if True, calculate the probability with add-one smoothing.
        :return: e -  a 2D numpy array of shape(#training_tags, #training_words_type)
        which contains the probabilities to assign word_i to a given tag_i:
        e[i1][i2] =P(word_i2|tag_i1) = Count(tag_i1, word_i2) / Count(tag_i1)
        and with add-one:            = (Count(tag_i1, word_i2)+1) / (Count(tag_i1) +V)
        where:
        V = current vocabulary length (distinct known and unknown words)
        i1 = np.where(order_training_tags == tag_i1)[0][0]
        i2 = np.where(order_known_words == word_i2)[0][0]
        """

        #number of distinct tags in training set
        num_tags = self.counts.order_training_tags.shape[0]
        #number of distinct words in training set
        num_words_type = self.counts.order_known_words.shape[0]

        e = np.empty((num_tags, num_words_type), dtype=float)

        for i in range(num_tags):
            tag = self.counts.order_training_tags[i]
            for j in range(num_words_type):
                word = self.counts.order_known_words[j]
                if (add_one):
                    e[i, j] = (self.counts.count_word_tag_pairs[(word, tag)] + 1) /\
                              (self.counts.count_training_tags[tag] + len(self.counts.known_words)
                                                                    + len(self.counts.unknown_words))
                else:
                    e[i, j] = self.counts.count_word_tag_pairs[(word, tag)] / self.counts.count_training_tags[tag]
        return e

    def find_index(self, list, token):
        """
        Given a numpy array of shape(N,) and a token,
        return the index of this token
        """
        return np.where(list == token)[0][0]

    def viterbi_algorithm(self, sentence, q, e):
        """
        Implementation of viterbi algorithm with bigram HMM tagger.
        This implementation chooses arbitrary tag from training set tags for unknown words.
        :param sentence: a list of words, which represents sentence/s.
        :param q: transition_probabilities.
        :param e: emission_probabilities.
        :return: predicted_tags - a list of tags.
                 predicted_tags[i] is the predicted tag for the word sentence[i].
        """

        # Initialization step
        T = len(sentence)
        K = self.counts.order_training_tags.shape[0]
        T1 = np.empty((K, T), 'd')
        T2 = np.empty((K, T), 'B')

        if sentence[0] in self.counts.known_words:
            index = self.find_index(self.counts.order_known_words, sentence[0])
            T1[:, 0] = 1*1*e[:, index]
        else:
            T1[:, 0] = 0
        T2[:, 0] = 0

        # Recursion step
        for i in range(1, T):
            if sentence[i] in self.counts.known_words:
                index = self.find_index(self.counts.order_known_words, sentence[i])
                T1[:, i] = np.max(T1[:, i - 1] * q.T * e[np.newaxis, :, index].T, 1)
            else:
                T1[:, i] = random.choice(np.arange(0, K)) # assign random tag from training set
            T2[:, i] = np.argmax(T1[:, i - 1] * q.T, 1)

        # Extract results
        result_tags = np.empty(T, 'B')
        result_tags[-1] = np.argmax(T1[:, T - 1])

        for i in reversed(range(1, T)):
            result_tags[i - 1] = T2[result_tags[i], i]

        final = self.counts.order_training_tags[result_tags[:]]

        return final

    def display_parameters_tables(self, trans_matrix, emmision_matrix):
        """
        Prints transition and emission matrices with names to each row and column.
        """
        t_df = pd.DataFrame(trans_matrix, columns=self.counts.order_training_tags, index=self.counts.order_training_tags)
        e_df = pd.DataFrame(emmision_matrix, columns=self.counts.order_known_words, index=self.counts.order_training_tags)
        print(t_df)
        print(e_df)

    def errors(self, training, test, add_one=False):
        """
        Calculates the error rate for viterbi algorithm.
        :param training: current training set.
        :param test: current test set.
        :param add_one: if True, calculate the emission probability with add-one smoothing.
        :return known_err, unknown_err, total_err
        """

        # convert test_set from sentences of tuples to single sequence of words
        test_tokens = [item[0] for item in test]

        trans_matrix = self.transition_probabilities(training)
        emission_matrix = self.emission_probabilities(add_one)

        tags_results = self.viterbi_algorithm(test_tokens, trans_matrix, emission_matrix)
        known_err, unknown_err, total_err = self.calculate_error_rate(tags_results, test)

        return known_err, unknown_err, total_err
