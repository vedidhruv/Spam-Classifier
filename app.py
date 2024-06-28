import streamlit as st
import pickle
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer

ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

st.title('Spam Ham Classifier')
st.write('This is a simple spam ham classifier using Naive Bayes')

input_text = st.text_input('Enter a message')

if st.button('Predict'):
    transform_input = transform_text(input_text)
    tfidf_text = tfidf.transform([transform_input])
    result = model.predict(tfidf_text)[0]

    if result == 1:
        st.write('Spam')
    else:
        st.write('Not Spam')
# Preprocess the input text
# 1. Lowercase
# 2. Remove special characters
# 3. Remove stopwords
# 4. Lemmatize
# 5. Vectorize
# 6. Predict
