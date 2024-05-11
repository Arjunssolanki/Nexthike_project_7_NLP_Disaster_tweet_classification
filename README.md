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

1. get_wordcounts(x):
2. get_charcounts(x):
3. get_avg_wordlength(x):
4. get_stopwords_counts(x):
5. get_hashtag_counts(x):
6. get_mentions_counts(x):
7. get_digit_counts(x):
8. get_uppercase_counts(x):
9. cont_exp(x):
10. get_emails(x):
11. remove_emails(x):
12. get_urls(x):
13. remove_urls(x):
14. remove_rt(x):
15. remove_special_chars(x):
16. remove_html_tags(x):
17. remove_accented_chars(x):
18. remove_stopwords(x):
19. make_base(x):
20. get_value_counts(df, col):
21. remove_common_words(x, freq, n=20):
22. remove_rarewords(x, freq, n=20):
23. remove_dups_char(x):
24. spelling_correction(x):
25. get_basic_features(df):
26. get_ngram(df, col, ngram_range):
27. get_word_frequency(tweet, column_name='text'):

## Notebook:

1. ### Disaster_tweet_classification.ipynb -

   #### Models used

   1. classification with TFIDF and SVM
   2. Classification with Word2Vec and SVM
   3. word embedding and classification with Deep learning

2. ### RF_Disaster_tweet_classification.ipynb -

   #### Model used

   1. Twitter Sentiment Analysis with Random Forest

3. ### TFIDF_Tweet_classification.ipynb

   10-fold cross validation

   #### models used:

   1. Multinomial Naive Bayes - TFIDF-Bigram
   2. Passive Aggressive Classifier - TFIDF-Bigram
   3. Multinomial Naive Bayes - TFIDF-Trigram
   4. Passive Aggressive Classifier - TFIDF-Trigram

4. ### Model folder:

   Saving the most accurate model using pickel in the folder name model which is we will use further for making the prediction and model deployment

5. ### app.py:

   This file consists of all the codes use to deploye the model using the flask and it use index.html saved in template.py to make connection with the flask app.

   To run the app.py exceute the below code in the terminal:

   ### python notebook/app.py
