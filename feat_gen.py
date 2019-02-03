#!/bin/python

from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import urllib2

brownClusterMap = {}
def preprocess_corpus(train_sents):

    target_url = "http://www.cs.cmu.edu/~ark/TweetNLP/clusters/750kpaths"
    data = urllib2.urlopen(target_url)
    for line in data:
        line = line.strip()
        cols = line.split("\t")
        brownClusterMap[cols[1]] = cols[0]

    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """
    pass

def token2features(sent, i, add_neighs = True):
    """Compute the features of a token.

    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that *fire*
    for the token. See the code below.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.

    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.

    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
    ftrs = []
    # bias
    ftrs.append("BIAS")
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent)-1:
        ftrs.append("SENT_END")

    # the word itself
    word = unicode(sent[i])
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())
    # some features of the word
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
    if word.isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")
    if word.lower().endswith("ing"):
        ftrs.append("IS_VERB")
    if len(word) >= 3:
        ftrs.append(word[-3:])
    if len(word) >= 2:
        ftrs.append(word[-2:])
    if word.lower().endswith("ed"):
        ftrs.append("IS_VERB")
    if word.lower().endswith("ies"):
        ftrs.append("IS_VERB")
    if word.lower().endswith("ly"):
        ftrs.append("IS_ADVERB")
    if word.lower().endswith("es"):
        ftrs.append("IS_NOUNS")
    if word.lower().endswith("ness"):
        ftrs.append("IS_NOUN")
    if word.lower().endswith("est"):
        ftrs.append("IS_ADJECTIVE")
    if word.lower().endswith("ous"):
        ftrs.append("IS_ADJECTIVE")
    if word.lower().endswith("ful"):
        ftrs.append("IS_ADJECTIVE")
    if word.lower().endswith("er"):
        ftrs.append("IS_ADJECTIVE")
    conjunction_list = ['and', 'nor', 'but', 'or', 'yet', 'so']
    if word.lower() in conjunction_list:
        ftrs.append("IS_CONJUNCTION")
    preposition_list = ['above', 'across', 'at', 'around', 'before', 'behind', 'below', 'between', 'beside', 'by', 'down', 'during', 'for', 'from', 'in', 'inside', 'onto', 'of', 'off', 'on', 'out', 'through', 'to', 'under', 'up', 'with']
    determinant_list = ['a', 'an', 'the', 'this', 'that', 'these', 'those']
    if word.lower() in preposition_list:
        ftrs.append("IS_PREPOSITION")
    if word.lower() in determinant_list:
        ftrs.append("IS_DETERMINANT")
    pronoun_list = ['I', 'him', 'her', 'it', 'me', 'you', 'his', 'their', 'they', 'you', 'your', 'them']
    if word.lower() in pronoun_list:
        ftrs.append("IS_PRONOUN")
    if word.startswith("@"):
        ftrs.append("IS_AT")
    if word.startswith("#"):
        ftrs.append("IS_HASHTAG")
    if word.startswith("http"):
        ftrs.append("IS_URL")
    if word.startswith("www."):
        ftrs.append("IS_URL")
    if word.lower().endswith(".com"):
        ftrs.append("IS_EMAIL")
    emoji_list = [':)', ':P', ':p', ':D', ':d', ':(', '<3' ,':*', ':O',  '</3', '=D', '-_-', ':@', ' =)', ';)']
    if word in emoji_list:
        ftrs.append("IS_EMOJI")
    if word in brownClusterMap:
        ftrs.append(brownClusterMap.get(word))

    lemmatizer = WordNetLemmatizer()
    if lemmatizer.lemmatize(word) not in ftrs:
        ftrs.append("Lemmatizer_" + lemmatizer.lemmatize(word))


    for set in wn.synsets(word):
        set = str(set).strip("Synset('")
        list = set.split(".")
        if list[0] not in ftrs:
            ftrs.append(list[0])

    #previous/next word feats
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i-1, add_neighs = False):
                ftrs.append("PREV_" + pf)
        if i < len(sent)-1:
            for pf in token2features(sent, i+1, add_neighs = False):
                ftrs.append("NEXT_" + pf)

    # return it!
    return ftrs

if __name__ == "__main__":
    sents = [
    [ "I", "love", "food" ]
    ]
    preprocess_corpus(sents)
    for sent in sents:
        for i in xrange(len(sent)):
            print sent[i], ":", token2features(sent, i)
