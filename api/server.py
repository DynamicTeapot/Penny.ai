# -*- coding: utf-8 -*-
# Author: DynamicTeapots

from multiprocessing.dummy import Pool as ThreadPool 
from flask import Flask
from flask import request
from flask import Response
from flask import json
from flask import jsonify
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, Normalizer
import wikipedia


app = Flask(__name__)

corpus =[]
categories = ['cooking', 'bathroom', 'sports', 'hardware', 'cars', 'technology', 'gardening', 'beauty', 'clothing']


def getWikiContent (topic): 
  return wikipedia.page(topic).content;

def getWikiSuggestions (topic):
  return wikipedia.search(topic)



# Plans to multithread this
def wikiIterator(topic, iterCount):
  if iterCount == None:
    iterCount = 2
  retval = ''
  current = [topic]
  todo = []
  while (iterCount > 0):
    iterCount=iterCount-1
    for item in current:
      print(item, iterCount)
      try:
        retval += getWikiContent(item)
      except:
        pass
      todo += getWikiSuggestions(item)
    print(iterCount, ' done')
    current = todo
    todo = []
  print(topic + ' researched.')
  return retval


tf = TfidfVectorizer(input='context', analyzer='word', ngram_range=(1, 5), lowercase=True, min_df=1, stop_words='english')

an = tf.build_analyzer();

for category in categories:
  corpus = corpus + [wikiIterator(category, 1)]

# Fit trains the vectorizer
# Transform looks for that
X = tf.fit_transform(corpus)

# Initialize the DBSCAN model which will both predict the data given and which we will train with fixed user input
# Valid values for metrics [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]
#Valid values for algorithm are {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}
# might want to change leaf size
# n_jobs is multithreading which takes up more space but since this will be ported to an ec2 it's alright for it to take up all the space
# might need to increase this eventually



km = KMeans(init='k-means++', algorithm='auto', max_iter=600, n_jobs=10)
db = DBSCAN(algorithm='auto', n_jobs='5')

km.fit_transform(X)
db.fit(Normalizer().fit_transform(X));

# This works really well
testSentence = 'Shrimp'
ts = tf.transform(testSentence)

print(km.predict(ts))


# request should be a jsonified string
# This should contain purely predicts methods
@app.route('/predict', methods=['POST'])
def predict():
  data = json.dumps(request.json)
  return km.predict(data)
  

# This should contain purely fits methods
@app.route('/train', methods={'POST'})
def train():
  resp = response(None, status=200)
  # Not quite sure how to train it with new data yet
  return resp

@app.route('/')
def landing():
  return 'Scikit learn categorizer'




