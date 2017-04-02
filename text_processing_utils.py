import re
from keras.preprocessing.sequence import pad_sequences
import numpy as np

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def vectorize_quries(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    for story, query in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        X.append(x)
        Xq.append(xq)
    return (pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen))

def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen), np.array(Y))

def read_story():
    story = []
    input_line = raw_input('story, empty to stop: ') 
    while input_line != '':
        # for now, lines have to be a full sentence
        if not input_line.endswith('.'):
           input_line += '.'
        story.extend(tokenize(input_line))
        input_line = raw_input('story, empty to stop: ') 
    return story
