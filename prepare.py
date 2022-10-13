import pandas as pd
import numpy as np

import unicodedata
import re
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords



def basic_clean(original):
    article = original.lower()
    
    article = unicodedata.normalize('NFKD', article).encode('ascii', 'ignore').decode('utf-8')
    
    article = re.sub(r'[^a-z0-9\'\s]', '', article)
    
    return article


def tokenize(article):
    tokenize = ToktokTokenizer()
    
    article = tokenize.tokenize(article, return_str=True)
    
    return article


def stem(article):
    
    ps = nltk.porter.PorterStemmer()
    
    stems = [ps.stem(word) for word in article.split()]
    
    article_stemmed = ' '.join(stems)
    
    return article_stemmed


def lemmatize(article):
    
    wnl = nltk.stem.WordNetLemmatizer()
    
    lemmas = [wnl.lemmatize(word) for word in article.split()]
    
    article_lemmatized = ' '.join(lemmas)
    
    return article_lemmatized


def remove_stopwords(article, extra_words = 'no_extra', exclude_words = 'all_good'):
        
    stopwords_ls = stopwords.words('english')
    
    if extra_words != 'no_extra':
        
        stopwords_ls = stopwords_ls.append(extra_words)
    else:
        stopwords_ls = stopwords_ls
            
    if exclude_words != 'all_good':
        
        stopwords_ls = stopwords_ls.remove(exclude_words)
    else:
        stopwords_ls = stopwords_ls
    
    words = article.split()
    
    filtered_words = [word for word in words if word not in stopwords_ls]
    
    article_without_stopwords = ' '.join(filtered_words)
    
    return article_without_stopwords