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
categories = ['Cooking', 'Bathroom', 'Sports', 'Hardware', 'Cars', 'Boats', 'Technology', 'Gardening', 'Beauty', 'Antiques', 'Clothing', 'Woodworking', 'Metalworking', 'Appliances' 'Housing', 'Furniture']


def getWikiContent (topic): 
  return wikipedia.page(topic).content;

def getWikiSuggestions (topic, breadth):
  return wikipedia.search(topic, results=breadth)

def naiveSuggestions (topic):
  return wikipedia.search(topic)




def wikerator(topic, depth=2, breadth=10):
  rows = []
  index = []
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
  data = data.append(wikerator(category, 2, 10))
# data = data.reset_index()
# data = data.reindex(np.random.permutation(data.index))
# Fit trains the vectorizer

# svd = TruncatedSVD()
# normalizer = Normalizer(copy=False)
# lsa = make_pipeline(svd, normalizer)

# X = lsa.fit_transform(X)

pipeline = Pipeline([
  ('vectorizer', TfidfVectorizer(input='context', analyzer='word', ngram_range=(1,6), lowercase=True, min_df=1, stop_words='english')),
  ('classifier', MultinomialNB())
  ])

# km = KMeans(init='k-means++', algorithm='auto', max_iter=600, n_jobs=10, n_clusters=2, verbose=True)
model = pipeline.fit(data['text'].values, data['class'].values)

print model.predict(['What a great flower garden temple waterfall flower flower garden grass weeds flowers flowers grass']);
# request should be a jsonified string
# Shold run a check to see if the document has enough length to proprely check it, and then specify 
# This should contain purely predicts methods
@app.route('/predict', methods=['POST'])
def predict():
  result = model.predict([request.form['data']])
  return result[0]

# This should contain purely fits methods
@app.route('/train', methods={'POST'})
def train():
  req = request.form['categories']
  resp = response(None, status=200)
  newData = DataFrame({'text': [], 'class': []})
  for category in req:
    newData = newData.append(wikerator(category, 1, 10))
  newData = newData.reindex(np.random.permutation(data.index))
  model.fit(newData['text'].values, newData['class'].values)
  return resp

@app.route('/')
def landing():
  return 'Item Categorizer'




