# SentimentAnalysis
First text mining project


Created by Koushik Sekar

Sentiment Analysis, IMDB Movie Reviews
Positive or Negative Response Prediction using Text Mining
Data Source - https://www.kaggle.com/c/word2vec-nlp-tutorial/data

Training data file (labeledTrainData.tsv) had 25,000 reviews
Imported in a Pandas dataframe, and split into two randomly - training dataset (80%) and testing dataset (20%)
Preprocessing was done to the review text values in the training dataset; and they were vectorized into bag of words.
Vocabulary was set to have 5000 most frequent words
Random Forest method was used to fit the model; Number of trees = 100

Preprocessing was also done on the testing dataset (n=5000); and it was vectorized like the training dataset
Model from the training dataset was used to predict the values on the testing dataset
The results were compared to the actual values in the testing dataset (See details in the sheet "Results Analysis Detail" in Sentiment Analysis Results Summary.xlsx)

Model consistently predicting the correct sentiment 84% of the time

Other details present in the code - sentiment_analysis.py
To run: python sentiment_analysis.py
