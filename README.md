## PROJECT OBJECTIVE :

### Building a machine learning model capable of accurately classifying tweets as either related to real disasters or not.

## My approach:

1. Importing the required libraries and reading the dataset
2. Exploratory Data Analysis (EDA)
3. Plotting wordcloud of Disaster Tweets
4. Data Cleaning by creating utils.py
5. tokenizing the text file and Training model
6. Evaluating the model and saving the most accurate model in folder model using pickle
7. Deploying the model using flask by creating a app.py and index.html

## Utils.py:

utils.py consist of all the function to cleane the text data set and find out functions to extract the important features from the data set:
get_wordcounts(x):  
get_charcounts(x):
get_avg_wordlength(x):
get_stopwords_counts(x):
get_hashtag_counts(x):
get_mentions_counts(x):
get_digit_counts(x):
get_uppercase_counts(x):
cont_exp(x):
get_emails(x):
remove_emails(x):
get_urls(x):
remove_urls(x):
remove_rt(x):
remove_special_chars(x):
remove_html_tags(x):
remove_accented_chars(x):
remove_stopwords(x):
make_base(x):
get_value_counts(df, col):
remove_common_words(x, freq, n=20):
remove_rarewords(x, freq, n=20):
remove_dups_char(x):
spelling_correction(x):
get_basic_features(df):
get_ngram(df, col, ngram_range):
get_word_frequency(tweet, column_name='text'):

## Notebook:

1. ### Disaster_tweet_classification.ipynb -

   #### Models used

   a. classification with TFIDF and SVM
   b. Classification with Word2Vec and SVM
   c. word embedding and classification with Deep learning

2. ### RF_Disaster_tweet_classification.ipynb -

   #### Model used

   a. Twitter Sentiment Analysis with Random Forest

3. ### TFIDF_Tweet_classification.ipynb

   10-fold cross validation

   #### models used:

   a. Multinomial Naive Bayes - TFIDF-Bigram
   b. Passive Aggressive Classifier - TFIDF-Bigram
   c. Multinomial Naive Bayes - TFIDF-Trigram
   d. Passive Aggressive Classifier - TFIDF-Trigram

4. ### Model folder:

   Saving the most accurate model using pickel in the folder name model which is we will use further for making the prediction and model deployment

5. ### app.py:

   This file consists of all the codes use to deploye the model using the flask and it use index.html saved in template.py to make connection with the flask app.

   To run the app.py exceute the above code in the terminal:
   python notebook/app.py
