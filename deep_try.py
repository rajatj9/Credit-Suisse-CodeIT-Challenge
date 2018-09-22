import logging
from codeitsuisse import  app
import itertools
import urllib.request
import json

#Imports for Deep learning
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
#Imports for Deep Learning trade here

from flask import request, jsonify;


@app.route('/', methods=['GET'])
def default_route():
    return "Team typhoont10 page";


@app.route('/machine-learning/question-1', methods=['POST'])
def prime_sum():
    data = request.get_json();
    X=np.array(data['input'])
    Y=np.array(data['input'])
    # Create linear regression object
    regr = linear_model.LinearRegression()
    
    # Train the model using the training sets
    regr.fit(X,Y)
    #coeffecients
    coeffecients= regr.coef_
    question=data['question']
    answer=0
    for i in range(len(coeffecients)):
        answer += coeffecients[i]*question[i]
    answer = np.sum(answer)
    print("answer = ", answer)
    ans_dict = {"answer" : answer }
    return json.dumps(ans_dict)
        