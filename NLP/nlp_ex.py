#                       #
# NLP using python NLTK #
#                       #

import pandas as pd
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer

messages = pd.read_csv('data\SMSSpamCollection',
                       sep='\t', names=['label', 'message'])
# print(messages.head(5))
# print(messages.describe())
print(messages.describe())
messages['length'] = messages['message'].apply(len)
# print(messages.head(5))
# plt.hist(x = 'length', bins = 60, data = messages)
# plt.show()


def process_text(mess):
    """
    1. Remove Punctuations
    2. Remove Stopwords
    3. Return the clean text
    """
    no_punch = [c for c in mess if c not in string.punctuation]
    no_punch = ''.join(no_punch)
    no_stopwords = [word for word in no_punch.split() if word.lower()
                    not in stopwords.words('english')]
    return no_stopwords


messages['message'] = messages['message'].apply(process_text)
print(messages.head(5))

bow = ''		# Bag of Words
try:
    bow = CountVectorizer(analyzer=process_text).fit(messages['message'])
except Exception as e:
    print('error in creating bow - > ', str(e))

print(len(bow.vocabulary_))
