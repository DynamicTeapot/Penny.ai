# -*- coding: utf-8 -*-
# Author: DynamicTeapots

from flask import Flask
from flask import request
from flask import Response
from flask import json
from flask import jsonify
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.preprocessing import scale


app = Flask(__name__)
tf = TfidfVectorizer(input='content', analyzer='word', ngram_range=[1, 5], lowercase=True, stop_words='english', norm='l2')

testCorpus = [
'A accelerator air bags air conditioner air filter air vent alarm antenna anti-lock brakes armrest auto automatic transmission automobile B babyseat back-up lights battery brake light brakes bumper C camshaft car carburetor chassis chrome trim clutch cooling system crankshaft cruise control D dashboard defroster diesel engine differential dimmer switch door door handle drive shaft E emergency brake emergency lights emissions engine exhaust system F fan belt fender floor mats frame fuel fuel gauge fuse G gas gas cap gasket gasoline gasoline engine gearbox gearshift gear stick glove compartment gps grille H hand brake headlight heater high-beam headlight hood horn hubcaps hybrid I ignition interior light internal combustion engine J jack K key L license plates lock low-beam headlight lugs M manifold manual transmission mat mirror motor mud flap muffler O odometer oil oil filter P parking lights passenger seat piston power brakes power steering R radiator radio rear-view mirror rear window defroster rims roof rotary engine S seat seat bags shock absorber side mirrors spare tire spark plug speedometer steering wheel suspension T tachometer tailgate thermometer tire trailer hitch trip computer trunk turbocharger turn signal U unleaded gas V vents visor W wheel wheel well windshield windshield wiper ',
'A antiseptic aspirin B bandages basin bath bath mat bath robe bath towel bathtub bidet brush bubble bath bubbles C cleaning cologne comb conditioner cotton balls curlers D dental floss disinfectant droppers dry E eyedropper F face cloth faucet floss flush G garbage can H hairbrush hair dryer hamper hand towel L laundry hamper lavatory loofah lotion M make-up medications medicine medicine cabinet mirror moisturizer mouthwash N nail clippers nail file nail scissors P paper towel perfume plumbing plunger powder Q Q-tips R razor razor blade restroom rug S scale scissors shampoo shave shaver shaving cream shower shower curtain shower stall sink soap soap dish soap dispenser sponge swabs T talcum power tissues toilet toilet paper toilet seat toothbrush toothpaste towel towel rack trash can tub tweezers U urinal W wash wash basin washroom waste basket water water closet WC whirlpool wipe',
'Third document',
'fourth document',
'etc.',
'highly technical thingy',
'more writing write write document files what what what what what',
'more editting documents what fun',
'document is unrelated']

# Fit trains the vectorizer
# Transform looks for that

X = tf.fit_transform(testCorpus)
# print(X)
# print(X)
# tf.fit_transform({'electronics macbook pro dell h ewlett packard': 'electronics'})

# Initialize the DBSCAN model which will both predict the data given and which we will train with fixed user input
# Valid values for metrics [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]
#Valid values for algorithm are {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}
# might want to change leaf size
# n_jobs is multithreading which takes up more space but since this will be ported to an ec2 it's alright for it to take up all the space
# might need to increase this eventually

# ftnames = tfidf_vectorizer.get_feature_names();

# db = DBSCAN(algorithm='auto', n_jobs='5')


km = KMeans(init='k-means++', algorithm='auto')

km.fit_transform(X)

testSentence = ['what']
ts = tf.transform(testSentence)

print(km.predict(ts))


# request should be a json object which when parsed contains an array of strings
# This should contain purely predicts methods
@app.route('/predict', methods=['POST'])
def predict():
  predictData = json.dumps(request.json)
  print(v)
  return

# This should contain purely fits methods
@app.route('/train', methods={'POST'})
def train():
  resp = response(None, status=200)
  return resp

@app.route('/')
def landing():
  return 'Scikit learn categorizer'




