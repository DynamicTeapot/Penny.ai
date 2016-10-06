# -*- coding: utf-8 -*-
# Author: DynamicTeapots

from pandas import DataFrame
from flask import Flask
from flask import request
from flask import Response
from flask import json
from flask import jsonify
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.naive_bayes import MultinomialNB
import wikipedia


app = Flask(__name__)

data = DataFrame({'text': [], 'class': []})
categories = ['cooking', 'bathroom', 'sports', 'hardware', 'cars', 'technology', 'gardening', 'beauty', 'clothing']


def getWikiContent (topic): 
  return wikipedia.page(topic).content;

def getWikiSuggestions (topic, breadth):
  return wikipedia.search(topic, results=breadth)

def naiveSuggestions (topic):
  return wikipedia.search(topic)




# Plans to multithread this
def wikiIterator(topic, depth, *breadth):
  if depth == None:
    depth = 2
  if breadth == None:
    breadth = 5
  retval = ''
  current = [topic]
  todo = []
  alltopics = [topic]
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
    alltopics = alltopics + todo
    todo = []
  print(topic + ' researched.')
  return {data: retval, topics: alltopics}


def wikerator(topic, depth, *breadth):
  rows = []
  index = []
  if depth == None:
    depth = 2
  if breadth == None:
    breadth = 5
  retval = ''
  current = [topic]
  todo = []
  alltopics = [topic]
  while (depth > 0):
    depth=depth-1
    for item in current:
      print item
      try:
        rows.append({'text': getWikiContent(item), 'class': topic})
        index.append(item)
      except:
        pass
      todo += getWikiSuggestions(item, breadth)
    print(depth, ' done')
    current = todo
    alltopics = alltopics + todo
    todo = []
  print(topic + ' researched.')
  return DataFrame(rows, index=index)




tf = TfidfVectorizer(input='context', analyzer='word', ngram_range=(1,6), lowercase=True, min_df=1, stop_words='english')

for category in categories:
  data = data.append(wikerator(category, 1, 10))
data = data.reindex(np.random.permutation(data.index))
print data
# Fit trains the vectorizer
# Transform looks for that
# X = tf.fit_transform(data['text'].values)

# svd = TruncatedSVD()
# normalizer = Normalizer(copy=False)
# lsa = make_pipeline(svd, normalizer)

# X = lsa.fit_transform(X)

pipeline = Pipeline([
  ('vectorizer', TfidfVectorizer(input='context', analyzer='word', ngram_range=(1,6), lowercase=True, min_df=1, stop_words='english')),
  ('classifier', MultinomialNB())
  ])

# km = KMeans(init='k-means++', algorithm='auto', max_iter=600, n_jobs=10, n_clusters=2, verbose=True)
print(pipeline.fit(data['text'].values, data['class'].values).predict(['cooking cars human what a wonderful world hahaha ofihwpidsghisept test che cut dice chop eat eat eat eat eat cooking chef cooking water boiling stove oven']));


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




