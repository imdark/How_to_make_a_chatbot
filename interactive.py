from text_processing_utils import *
from memory_network import *
import numpy as np


memory_network = MemoryNetwork()

while True:
    print 'Please only use this vocabulary:\n' + ' , '.join(memory_network.word_idx.keys())
    story = read_story()
    print 'Story: ' + ' '.join(story)
    question = raw_input('q:')
    if question == '' or question == 'exit':
        break
    story_vector, query_vector = vectorize_quries([(story, tokenize(question))],
                                    memory_network.word_idx, 68, 4)
    prediction = memory_network.model.predict([np.array(story_vector), np.array(query_vector)])
    prediction_word_index = np.argmax(prediction)
    for word, index in memory_network.word_idx.items():
        if index == prediction_word_index:
            print word
