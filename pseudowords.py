import re
from word_counting import WORD, TAG

def detect_suffix(word):
    """
    Return the suffix of a given word.
    Suitable longer suffix will be favorite over suitable shorter suffix, because it
    captures the spelling more correctly.
    :param word: a string of length >= 2
    :return: a uppercase string which is a relevant suffix of the word.
             if there is no match, return empty string.
    """
    bi_suffix = {'ed', 'er', '\'s', 'es', 'ly', 'er', 'en', 'el', 'al', 'ic', 'th', 'or'}
    tri_suffix = {'ive', 'acy', 'ate', 'ify', 'ble', 'ous', 'ent', 'ant', 'est', 'ish',
                  'ful', 'age', 'ity', 'dom', 'ism', 'ist', 'ian', 'ess', 'ize', 'ite', 'ing',
                  'ale', 'ile', 'ogy', 'ers', 'ele', 'ane', 'ows', 'ine'}
    quad_suffix = {'ance', 'ible', 'ment', 'less', 'ness', 'ence', 'tude', 'tion', 'sion',
                   'hood', 'ship', 'ions', 'wood', 'port', 'main'}

    if len(word) >= 4:
        if (word[-4] + word[-3] + word[-2] + word[-1]) in quad_suffix:
            return (word[-4]+word[-3]+word[-2]+word[-1]).upper()

    if len(word) >= 3:
        if (word[-3] + word[-2] + word[-1]) in tri_suffix:
            return (word[-3]+word[-2]+word[-1]).upper()

    if len(word) >= 2:
        if (word[-2] + word[-1]) in bi_suffix:
            return (word[-2]+word[-1]).upper()

    return ''

def detect_prefix(word):
    """
    Return the prefix of the given word.
    Suitable longer prefix will be favorite over suitable shorter prefix, because it
    captures the spelling more correctly.
    :param word: a string of length >= 2
    :return: a uppercase string which is a relevant prefix of the word.
             if there is no match, return empty string.
    """
    bi_prefix = {'un', 'im', 'in', 'ir', 'il', 'de', 'ab', 'ex', 're','bi', 'co', 'en', 'em', 'be'}
    tri_prefix = {'dis', 'non', 'mis', 'mal', 'out', 'sub', 'pre', 'uni', 'tri', 'pro'}
    quad_prefix = {'anti', 'over', 'hypo', 'fore', 'post', 'mono', 'poly', 'omni', 'phil', 'bene',
                   'ambi', 'homo', 'auto', 'circ'}

    if len(word) >= 4:
        if (word[0] + word[1] + word[2] + word[3]) in quad_prefix:
            return (word[0]+word[1]+word[2]+word[3]).upper()

    if len(word) >= 3:
        if (word[0] + word[1] + word[2]) in tri_prefix:
            return (word[0]+word[1]+word[2]).upper()

    if len(word) >= 2:
        if (word[0] + word[1]) in bi_prefix:
            return (word[0]+word[1]).upper()

    return ''

def detect_numbers(word):
    """
    Checks if the given input contains one or more numbers.
    """
    if(re.match("[0-9]+", word[WORD])):
        return True
    return False

def pseudowords_mapper(word):
    """
    Maps given word to a pseudo-word, by considering spelling factors.
    """

    capitals = {'A','B','C','D','E','F','G','H','I','J','K','L','M',
                'N','O','P','Q','R','S','T','U','V','W','X','Y','Z'}

    if '$' in word[WORD]:
        return '_MONEY_'

    if detect_numbers(word):
        return "_NUMBERS_"

    if len(word[WORD]) >= 2:
        suffix = detect_suffix(word[WORD])
        prefix = detect_prefix(word[WORD])

        if suffix != '' or prefix != '':
            return prefix + "_" + suffix

    if word[WORD][0] in capitals:
        return "_CAPITAL_"

    if word[WORD][-1] == 's':
        return '_S_'

    if word[WORD][-1] == 'y':
        return '_Y_'

    if '-' in word[WORD]:
        return '_-_'

    if '\'' in word[WORD]:
        return '_\'_'

    return "N\A"

def replace_to_pseudowords(training, test, counts, threshold=5):
    """
    Return modified flat training and test set as follows:
    All non-frequency words from training set, and all unknown words (from test set)
    will map to pseudo-words.
    *frequency words = words occurring >= threshold times in training set.
    """

    # words that appear strictly less than the threshold
    low_frequency_words = set(item for item in counts.count_training_words
                              if counts.count_training_words[item] < threshold)

    # map low-frequency words to their pseudo-words
    new_training_set = training.copy()
    for i, word in enumerate(new_training_set):
        if word[WORD] in low_frequency_words:
            new_training_set[i] = (pseudowords_mapper(word), word[TAG])

    # map unknown-words to their pseudo-words
    new_test_set = test.copy()
    for i, word in enumerate(new_test_set):
        if word[WORD] not in counts.known_words:
            new_test_set[i] = (pseudowords_mapper(word), word[TAG])

    return new_training_set, new_test_set
