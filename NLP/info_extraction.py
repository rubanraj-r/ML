# import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

ps = PorterStemmer()
utterance = input('Input your text - > ')


def process_text(text):
    '''
    1. Tokenizing
    2. Removing Punctuations
    3. Remove Stopwords
    4. Normalizing using Porter Stemmer
    5. Parts of Speech tagging
    6. Removing chunks
    '''
    try:
        tokened = nltk.word_tokenize(utterance)
        no_punc_utter = [c for c in tokened if c not in string.punctuation]
        no_punc_utter = ' '.join(no_punc_utter)
        no_stopwords = [c for c in no_punc_utter.split() if c.lower()
                        not in stopwords.words('english')]
        # stemmed = [ps.stem(c) for c in no_stopwords]
        tagged = nltk.pos_tag(no_punc_utter)
        chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
        chunkParser = nltk.RegexpParser(chunkGram)
        chunked = chunkParser.parse(tagged)
        # chunked.draw()
        return chunked

    except Exception as e:
        print('Error in processing text - > ', str(e))
        return 'process failed'


processed_uttr = process_text(utterance)
print(processed_uttr)
text = nltk.Text(processed_uttr)
print(text.similar('woman'))
