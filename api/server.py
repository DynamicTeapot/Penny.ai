# -*- coding: utf-8 -*-
# Author: DynamicTeapots

from multiprocessing.dummy import Pool as ThreadPool 
from flask import Flask
from flask import request
from flask import Response
from flask import json
from flask import jsonify
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import wikipedia


app = Flask(__name__)

corpus =[]
categories = ['cooking', 'bathroom', 'sports', 'hardware', 'cars', 'technology', 'gardening', 'beauty', 'clothing']


def getWikiContent (topic): 
  return wikipedia.page(topic).content;

def getWikiSuggestions (topic, breadth):
  return wikipedia.search(topic, results=breadth)



# Plans to multithread this
def wikiIterator(topic, depth, *breadth):
  if depth == None:
    depth = 2
  if breadth == None:
    breadth = 5
  retval = ''
  current = [topic]
  todo = []
  while (depth > 0):
    depth=depth-1
    for item in current:
      print item
      try:
        retval += getWikiContent(item)
      except:
        pass
      todo += getWikiSuggestions(item, breadth)
    print(depth, ' done')
    current = todo
    todo = []
  print(topic + ' researched.')
  return retval


tf = TfidfVectorizer(input='context', analyzer='word', ngram_range=(1,6), lowercase=True, min_df=1, stop_words='english')

for category in categories:
  corpus = corpus + [wikiIterator(category, 2, 5)]

# Fit trains the vectorizer
# Transform looks for that
X = tf.fit_transform(corpus)

svd = TruncatedSVD()
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(X)

print(X)

# Initialize the DBSCAN model which will both predict the data given and which we will train with fixed user input
# Valid values for metrics [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]
#Valid values for algorithm are {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}
# might want to change leaf size
# n_jobs is multithreading which takes up more space but since this will be ported to an ec2 it's alright for it to take up all the space
# might need to increase this eventually



km = KMeans(init='k-means++', algorithm='auto', max_iter=600, n_jobs=10, n_clusters=len(categories), verbose=True)

km.fit(X)
# This works really well with small documents?
# ts = tf.fit_transform(['Cooking Cooking Cooking'])

test = tf.fit_transform(['cooking pots pans running water marble stove fire heat above 220 degrees cooking cooking chef chef chef', '', '', '', '', '', '', '', '', '']);

test = lsa.fit_transform(test)

print(km.predict(test))

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
  return 'Item Categorizer'




