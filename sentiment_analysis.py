# Created by Koushik Sekar
# Text mining project  - Sentiment analysis

# Import NLP and related packages
import pandas as pd						# Import the pandas package
from bs4 import BeautifulSoup					# Import BeautifulSoup into your workspace
import re							# To use regular expressions and replace
import nltk							# Import Natural Language Processing Toolkit
from nltk.corpus import stopwords				# To use stopwords from nltk
import numpy as np						# Import Numpy
from sklearn.ensemble import RandomForestClassifier		# Import Random Forest
from sklearn.feature_extraction.text import CountVectorizer	# To vectorize

train_init = pd.read_csv("labeledTrainData.tsv",header=0,delimiter="\t",quoting=3)

# Form training and testing dataset
train = train_init.sample(frac=0.8,random_state=200)
test = train_init.drop(train.index)

# Delete the sentiment column for the testing dataset
del test["sentiment"]

def review_to_words ( raw_review ):
# Function to convert a raw review to a string of words
# The input is a single string (a raw movie review), and the output is a single string (a preprocessed movie review)
#
# 1. Remove HTML
  review_text = BeautifulSoup(raw_review).get_text()
#
# 2. Remove Non-letters
  letters_only = re.sub("[^a-zA-Z]"," ",review_text)
#
# 3. Convert to lower case, split into individual words
  words = letters_only.lower().split()
#
# 4. Convert the stop words to a set
  stops = set (stopwords.words("english"))
#
# 5. Remove stop words
  meaningful_words = [w for w in words if not w in stops]
#
# 6. Join the words back into one string separated by a space, and return the result
  return(" ".join( meaningful_words ))

train = train.reset_index()

# Steps to loop through and clean all the reviews in the dataset
# 1. Get a count of the number of reviews in the dataset
num_reviews = train["review"].size

# 2.Initialize an empty list to hold all the clean reviews
clean_train_reviews = []

#3. Loop over each review, create an index to go from 0 to the length of the review list
for i in xrange(0, num_reviews):
  clean_train_reviews.append( review_to_words( train["review"][i] ))

print "Creating the bag of words...\n"

# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)

# Use the "fit_transform" function to fit the model and learn the vocabulary. It also transforms data to faeture vectors
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Convert the results to a numpy array
train_data_features = train_data_features.toarray()

vocab = vectorizer.get_feature_names()

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

print "Training the random forest..."

# Initialize the random forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)

forest = forest.fit( train_data_features, train["sentiment"] )

# The RandomForest Model has now been trained, should now test it on the testing dataset
test = test.reset_index()

# Create an empty list and append the clean reviews one by one
num_reviews = len(test["review"])
clean_test_reviews = []

print "Cleaning and parsing the test set movie reviews...\n"
for i in xrange(0,num_reviews):
  clean_review = review_to_words( test["review"][i] )
  clean_test_reviews.append( clean_review )

# Get a bag of words from a test set and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use Random Forest to make sentimental label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the CSV output file
output.to_csv( "Bag_of_Words_Results.csv", index=False, quoting=3 )
