# -*- coding: utf-8 -*-
# Author: DynamicTeapots

from flask import Flask
from flask import request
from flask import Response
from flask import json
from flask import jsonify
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction import DictVectorizer
app = Flask(__name__)

# High level Abstraction:
# This is a vectorizer which converts words to numbers so we can run machine learning algorithms on them
v = DictVectorizer(sparse = True)
# Run v.fitTransform(List[Dicts])

# Initialize the DBSCAN model which will both predict the data given and which we will train with fixed user input
# Valid values for metrics [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]
#Valid values for algorithm are {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}
# might want to change leaf size
# n_jobs is multithreading which takes up more space but since this will be ported to an ec2 it's alright for it to take up all the space
# might need to increase this eventually
DBSCAN(metric='euclidean', algorithm='auto', n_jobs='5')

@app.route('/predict', methods=['POST'])
def predict():
  predictData = json.dumps(request.json)
  print(v)
  return
  #request should be a json object with certain characteristics


@app.route('/train', methods={'POST'})
def train():
  resp = response(None, status=200)
  return resp

@app.route('/')
def landing():
  return 'Scikit learn categorizer'