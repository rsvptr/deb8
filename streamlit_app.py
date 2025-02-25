# Importing required libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy import sparse
import nltk
import re
import string
import warnings

from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize  # (if needed later)
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

warnings.filterwarnings('ignore')

# Ensure necessary NLTK data is available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Set Streamlit page configuration
st.set_page_config(page_title="Deb8: Clickbait Detector", layout="centered")

# CSS for Streamlit webpage background
def set_bg_hack_url():
    st.markdown(
         """
         <style>
         .stApp {
             background: url("https://images.unsplash.com/photo-1613778081725-c00e8943980b?q=80");
             background-size: cover;
         }
         </style>
         """,
         unsafe_allow_html=True
     )

# Apply background styling
set_bg_hack_url()

# Adding headings to the webpage
st.markdown("# Deb8 🎣⚔️")
st.markdown("This interactive dashboard is designed to assess any article headline and determine if it is clickbait or not. To evaluate a headline, simply enter it below and click 'Submit'.")

# Load the pre-trained model and the TF-IDF vectorizer using context managers
with open('Model and Vectorizer/naive-bayes_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('Model and Vectorizer/tf-idf_vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Load stopwords list
stopwords_list = stopwords.words('english')

# Define functions for text preprocessing and feature engineering
def clean_text_round1(text):
    """
    Cleans the input text by converting it to lowercase, removing URLs, newline characters,
    multiple spaces, punctuation, and certain special characters.
    """
    text = text.lower()
    text = re.sub('\n', ' ', text)
    text = re.sub('  ', ' ', text)
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
    text = re.sub(r'\[.*?\]', ' ', text)  # Use a raw string to avoid escape issues
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('“', '', text)
    text = re.sub('”', '', text)
    text = re.sub('’', '', text)
    text = re.sub('–', '', text)
    text = re.sub('‘', '', text)
    return text

def contains_question(headline):
    """Returns 1 if the headline contains a question mark or starts with common interrogative words; otherwise 0."""
    if "?" in headline or headline.startswith(('who','what','where','why','when','whose','whom','would','will','how','which','should','could','did','do')):
        return 1
    return 0

def contains_exclamation(headline):
    """Returns 1 if the headline contains an exclamation mark; otherwise 0."""
    return 1 if "!" in headline else 0

def starts_with_num(headline):
    """Returns 1 if the headline starts with a number; otherwise 0."""
    return 1 if headline and headline[0].isdigit() else 0

# Create an area in the Streamlit app for the user to input a headline
sentence = st.text_area('Enter headline here')

# Processing the submitted headline upon button click
if st.button('Submit'):
    # Clean and engineer features from the headline
    cleaned_sentence = clean_text_round1(sentence)
    headline_words = len(cleaned_sentence.split())
    question_feature = contains_question(cleaned_sentence)
    exclamation_feature = contains_exclamation(cleaned_sentence)
    starts_with_num_feature = starts_with_num(cleaned_sentence)
    
    # Transform the cleaned headline using the pre-loaded vectorizer
    input_text = [cleaned_sentence]
    vectorized = vectorizer.transform(input_text)
    
    # Convert numeric features into a sparse matrix
    numeric_features = np.array([[question_feature, exclamation_feature, starts_with_num_feature, headline_words]])
    numeric_sparse = sparse.csr_matrix(numeric_features)
    
    # Combine numeric features with the TF-IDF features
    final_features = sparse.hstack([numeric_sparse, vectorized])
    
    # Predict using the loaded model
    result = model.predict(final_features)
    if result == 1:
        st.error('🚨 Alert: Clickbait Detected! 🚨 This headline appears to be crafted to lure clicks. Proceed with caution.')
    else:
        st.success('🌟 No Clickbait Here! 🌟 This headline seems genuine and straightforward. Happy reading!')
