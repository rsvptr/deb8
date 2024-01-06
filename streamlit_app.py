# Importing required libraries
import streamlit as st  # Streamlit library for building web apps
import pandas as pd  # Pandas for data manipulation
import numpy as np  # Numpy for numerical operations
import pickle  # For loading serialized Python object files
from pickle import load  # Specific function for loading pickle files
from scipy import sparse  # For handling sparse matrices
import nltk  # Natural Language Toolkit for text processing
nltk.download('stopwords')  # Downloading stopwords for text cleaning
from nltk.corpus import stopwords  # Import stopwords
from nltk.tokenize import word_tokenize  # Tokenizer
from nltk.probability import FreqDist  # Frequency distribution
from nltk.stem import PorterStemmer  # Stemming
import re  # Regular expression operations
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes classifier
import string  # String operations

# Importing text feature extraction tools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import warnings
warnings.filterwarnings('ignore')  # Ignoring warnings for cleaner output

# CSS for Streamlit webpage background
def set_bg_hack_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://images.unsplash.com/photo-1518655048521-f130df041f66");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Applying the background to the Streamlit webpage
set_bg_hack_url()

# Adding headings to the webpage
st.markdown("# 🎣 Clickbait Headline Detector 🔍")
st.markdown("This interactive dashboard is designed to assess any article"
            " headline and determine if they are clickbait or not. To evaluate"
            " a headline, simply enter it in the space provided below and click"
            " 'Submit' for an instant analysis.")

# Loading the pre-trained model and the TfidfVectorizer from serialized files
model = pickle.load(open('Model and Vectorizer/naive-bayes_model.pkl','rb'))  # Load Naive Bayes model
stopwords_list = stopwords.words('english')  # Load stopwords list
vectorizer = load(open('Model and Vectorizer/tf-idf_vectorizer.pkl','rb'))  # Load TF-IDF vectorizer

# Define functions for text preprocessing and feature engineering
def clean_text_round1(text):
    '''Cleans the input text by converting to lowercase, removing URLs, square brackets, punctuation, and unnecessary characters.'''
    # Various text cleaning steps
    text = text.lower()
    text = re.sub('\n', ' ', text)
    text = re.sub('  ', ' ', text)
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('“','',text)
    text = re.sub('”','',text)
    text = re.sub('’','',text)
    text = re.sub('–','',text)
    text = re.sub('‘','',text)
    return text

# Functions to identify specific features in the headlines
def contains_question(headline):
    '''Checks if the headline contains a question.'''
    if "?" in headline or headline.startswith(('who','what','where','why','when','whose','whom','would','will','how','which','should','could','did','do')):
        return 1
    else:
        return 0

def contains_exclamation(headline):
    '''Checks if the headline contains an exclamation mark.'''
    if "!" in headline:
        return 1
    else:
        return 0

def starts_with_num(headline):
    '''Checks if the headline starts with a number.'''
    if headline.startswith(('1','2','3','4','5','6','7','8','9')):
        return 1
    else:
        return 0

# Creating an area in the Streamlit app for the user to submit a headline
sentence = st.text_area('Enter headline here')

# Processing the submitted headline upon button click
if st.button('Submit'):
    # Cleaning and feature engineering on the submitted headline
    cleaned_sentence = clean_text_round1(sentence)
    headline_words = len(cleaned_sentence.split())
    question = contains_question(cleaned_sentence)
    exclamation = contains_exclamation(cleaned_sentence)
    starts_with_num = starts_with_num(cleaned_sentence)
    input = [cleaned_sentence]
    vectorized = vectorizer.transform(input)
    final = sparse.hstack([question,exclamation,starts_with_num,headline_words,vectorized])
    
    # Using the model to predict and display the result
    result = model.predict(final)
    if result == 1:
        st.error('🚨 Alert: Clickbait Detected! 🚨 This headline appears to be crafted to lure clicks. Proceed with caution.')
    else:
        st.success('🌟 No Clickbait Here! 🌟 This headline seems genuine and straightforward. Happy reading!')
