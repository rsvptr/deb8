# Deb8: An NLP and ML-based Clickbait Detector

## Table of Contents
1. [Project Description](#project-description)
2. [Objective](#objective)
3. [Methodology](#methodology)
4. [Dataset Information](#dataset-information)
5. [Steps Involved](#steps-involved)
   - [Data Scraping](#data-scraping)
   - [Data Processing & Feature Engineering](#data-processing-and-feature-engineering)
   - [Exploratory Data Analysis](#exploratory-data-analysis)
   - [Modeling & Evaluation](#modeling-and-evaluation)
6. [Directory Structure](#directory-structure)
7. [Deployed Model](#deployed-model)
8. [Conclusion](#conclusion)
9. [Future Work](#future-work)
10. [Additional Resources](#additional-resources)
11. [Contributors](#contributors)

---

<a name="project-description"></a>
# Project Description
Deb8 is a project aimed at distinguishing clickbait headlines from non-clickbait ones using Natural Language Processing (NLP) and Machine Learning (ML). It leverages various text classification techniques such as n-grams, TF-IDF, tokenization, and stopword removal, coupled with ML models to categorize headlines. This project also showcases the potential of scaling such a model for business use-cases like reducing clutter on social media and other internet platforms.

<a name="objective"></a>
# Objective
The primary goal of Deb8 is to provide a tool that can accurately determine whether a given headline is clickbait or not. It aims to aid in better information consumption by allowing users to check headlines, thereby promoting more authentic and trustworthy content circulation.

<a name="methodology"></a>
# Methodology
The project follows a comprehensive approach, incorporating data collection, cleaning, processing, exploratory analysis, and modeling. It applies NLP techniques like TF-IDF, tokenization, and stopword removal, alongside models like Naive Bayes, Logistic Regression, and SVM for effective classification.

<a name="dataset-information"></a>
## Dataset Information

The dataset used in the Deb8 project is a crucial component for training the clickbait detection model. It comprises a wide array of headlines gathered from various sources, categorized into two main types: clickbait and non-clickbait. Below is a detailed breakdown of the dataset:

### Sources and Composition

1. **Kaggle Dataset (2007-2016)**: 
   - Source: [Kaggle](https://www.kaggle.com/amananandrai/clickbait-dataset)
   - Author: Aman Anand Rai
   - Headlines: 30,000
   - Period: 2007 to 2016
   - Nature: Exclusively clickbait headlines.

2. **Custom Scraped Dataset (2019-2020)**:
   - Headlines: 22,000
   - Period: 2019 to 2020
   - Sources: Twitter, various APIs, online publications.
   - Inclusion: 
     - Clickbait sources such as Thatscoop, Viralstories, Political Insider, Examiner, The Odyssey, Buzzfeed, Upworthy, Viral Nova, Bored Panda.
     - Non-clickbait sources like The Guardian, Bloomberg, The Hindu, The New York Times, The Washington Post, Wiki News, Reuters.

### Structure and Features

- The dataset predominantly contains headline text, classified into clickbait and non-clickbait categories.
- Supplementary features developed through feature engineering include:
  - **headline_words**: The total number of words in a headline.
  - **question**: Indicates if a headline begins with a question word or includes a '?'.
  - **exclamation**: Shows if a headline contains an exclamation mark.
  - **starts_with_num**: Flags headlines beginning with a numeral.

<a name="steps-involved"></a>
## Steps Involved

### Data Scraping

The first phase in the data collection process involved developing scripts to scrape headlines from a variety of sources. This included:

- **Twitter**: Using custom scrapers to gather headlines from specific Twitter accounts known for clickbait content.
- **Online Publications**: Utilizing APIs from various news sources to fetch headlines. These sources included reputable outlets for non-clickbait headlines and popular clickbait sources.
- **Tools and Libraries**: Scripts were written in Python, leveraging libraries such as BeautifulSoup for web scraping and requests for API interactions.

### Data Processing and Feature Engineering

Once the data was collected, the following processing steps were undertaken:

- **Cleaning**: The raw headlines were cleansed of any unnecessary characters such as punctuation marks, URLs, and non-textual elements to ensure the quality of the text data.
- **Stopword Removal**: Standard NLP practices were applied to remove common English stopwords.
- **Tokenization**: Headlines were broken down into tokens (words) to facilitate further analysis.
- **Feature Engineering**: Additional features were created to enhance the model's learning capability, including:
  - **Word Count** (`headline_words`): The number of words in each headline.
  - **Question Presence** (`question`): Indicator of whether the headline contains a question.
  - **Exclamation Mark Presence** (`exclamation`): Detection of exclamation marks within the headline.
  - **Number at Start** (`starts_with_num`): Whether the headline begins with a number.

### Exploratory Data Analysis

In this stage, the dataset underwent a thorough analysis to uncover patterns and insights:

- **Word Frequency Analysis**: Investigating the most common words in clickbait and non-clickbait headlines.
- **Feature Distribution**: Analyzing how the newly engineered features like word count, questions, and exclamation marks varied across clickbait and non-clickbait categories.
- **Visualization**: Utilizing plots and charts to visually represent these distributions and frequencies for better understanding and insights.

### Modeling and Evaluation

The core of the project focused on developing and evaluating various machine learning models:

- **Model Selection**: The models included Naive Bayes, Random Forest, Linear SVM, Logistic Regression, and XGBoost.
- **Training**: Each model was trained on the processed dataset, learning to differentiate between clickbait and non-clickbait headlines based on the features.
- **Evaluation Metrics**: The models' performances were primarily assessed using accuracy and recall metrics to ensure a balanced perspective on their predictive abilities.
- **Model Tuning**: Based on initial results, models were fine-tuned for optimal performance by adjusting parameters and refining the feature set.


<a name="directory-structure"></a>
# Directory Structure
```
Deb8
│
├── LICENSE
│   └── [License file detailing the project's licensing terms]
│
├── Procfile
│   └── [Specifies commands that are executed by the app on startup]
│
├── requirements.txt
│   └── [List of Python dependencies required for the project]
│
├── runtime.txt
│   └── [Specifies the Python runtime version for the project]
│
├── setup.sh
│   └── [Shell script for setting up environment variables and configurations]
│
├── streamlit_app.py
│   └── [Main Streamlit application script for the web interface]
│
├── .streamlit
│   └── config.toml
│       └── [Configuration settings for Streamlit application]
│
├── Datasets
│   ├── 1 - Scraped Datasets
│   │   └── [Folder containing CSV files of scraped titles from various sources]
│   │
│   └── 2 - Processed Datasets
│       └── [Folder containing processed datasets post-cleaning and feature engineering]
│
├── Model and Vectorizer
│   ├── naive-bayes_model.pkl
│   │   └── [Saved Naive Bayes model after training]
│   │
│   └── tf-idf_vectorizer.pkl
│       └── [Saved TF-IDF vectorizer used in feature extraction]
│
└── Source Code
    ├── 1 - Scraping
    │   └── [Scripts and Jupyter notebooks used for scraping headlines from various sources]
    │
    ├── 2 - Processing
    │   └── [Scripts for data cleaning, preprocessing, and feature engineering]
    │
    └── 3 - Modeling
        └── [Scripts and notebooks for building, training, and evaluating ML models]
```

<a name="deployed-model"></a>
# Deployed Model

The project culminates in an interactive web application, built using Streamlit, designed for real-time clickbait detection. Here's an overview of how the deployed model functions and the user experience it offers:

### Web Interface and User Experience

- **Streamlit App**: The core of the user interface is a Streamlit web application, making it lightweight yet powerful.
- **Background and Styling**: The app features a custom background and CSS styling for an engaging user experience.
- **Heading and Description**: At the top, users are greeted with a clear heading and a brief description of the app's purpose and functionality.

### Headline Submission and Analysis

- **Input Area**: Users can enter any headline into a text area. This flexibility allows for a wide range of headline styles and sources.
- **Submit Button**: After entering the headline, users can submit it for analysis with a simple click.

### Behind-the-Scenes Processing

- **Pre-Trained Model**: The app utilizes a pre-trained Naive Bayes model, ensuring quick and reliable predictions.
- **Text Preprocessing**: Submitted headlines undergo thorough cleaning and preprocessing, similar to the training data. This includes lowercasing, URL removal, punctuation removal, and more.
- **Feature Engineering**: The app applies the same feature engineering techniques used in model training, including checking for questions, exclamation marks, and numerical beginnings.
- **TF-IDF Vectorization**: Headlines are transformed using the pre-loaded TF-IDF vectorizer, aligning them with the model's expected input format.

### Prediction and Feedback

- **Instant Analysis**: The model quickly analyzes the headline and predicts whether it is clickbait or not.
- **User Feedback**: The app provides immediate feedback:
  - If clickbait is detected, it displays an alert warning the user about potential clickbait.
  - If no clickbait is detected, it reassures the user of the headline's genuineness.

### Accessibility

- **Web-Based Platform**: Being web-based, the app is accessible from any device with internet access, requiring no additional software installation.
- **Live Version**: The deployed model is hosted online, and users can access it [here.](#)


<a name="conclusion"></a>
# Conclusion
Deb8 successfully leverages machine learning and NLP techniques to distinguish between clickbait and non-clickbait headlines with high accuracy and recall. It demonstrates the feasibility of scaling such a solution for broader, real-world applications.

<a name="future-work"></a>
# Future Work
- Exploring deep learning and neural network models for potentially more robust classification.
- Investigating LDA for topic analysis and its use in modeling.
- Testing the model on newer datasets for continuous improvement.

<a name="additional-resources"></a>
# Additional Resources
A detailed report on the project's methodology, findings, and insights is available in the repository for further reference. 

<a name="contributors"></a>
# Contributors
 
 - Romy Savin Peter (S20190010153)  
 - Krushang Sirikonda (S20190010164)
 - Emma Mary Cyriac (S20190010048)    
 - Riya Rajesh (S20190010152)

