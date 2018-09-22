import logging
from codeitsuisse import  app
import itertools
import urllib.request
import json
import copy

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
        

        
@app.route('/tally-expense', methods=['POST'])
def getOut():
    Data = request.get_json();
    print(Data)
    n=len(Data['persons'])
    Person=Data['persons']
    Expense=Data['expenses']
    DictName={}
    for p in range(n):
        DictName[p]=Data['persons'][p]
    def getTotal(Data):
        total=0
        nameDict={}
        for z in range(n):
            nameDict[Person[z]]=0
        for i in range(n):
            now=Expense[i]
            nameDict[now['paidBy']]-=now['amount']
            if 'exclude' in now.keys():
                PersonName=[]
                for p in range(n):
                    if now['exclude'].count(Person[p])==0:
                        PersonName.append(Person[p])
                for d in range(len(PersonName)):
                    nameDict[PersonName[d]]+=(now['amount']/(n-len(PersonName)))
            else:total+=now['amount']    
        for k in range(n):
            nameDict[Person[k]]+=(total/n)
        return nameDict
    nameDictTotal=getTotal(Data)
    total=[]
    for d in range(n):
        total.append(nameDictTotal[DictName[d]])
    Trans=[]
    #temp=copy.copy(total)
    while total!=[0]*n:
        temp=copy.copy(total)
        step={'from':0,'to':0,'amount':0}
        tempMin=temp[total.index(min(total))]
        tMin=tempMin
        tempMax=temp[total.index(max(total))]
        tMax=tempMax
        if(tempMax+tempMin)>0:
            tMax=tempMax+tempMin
            tMin=0
            minIndex=total.index(min(total))
            maxIndex=total.index(max(total))
            temp[minIndex]=tMin
            temp[maxIndex]=tMax
            step['from']=DictName[maxIndex]
            step['to']=DictName[minIndex]
            step['amount']=round(min(tempMax,abs(tempMin)),2)
            Trans.append(step)
            total=temp
        elif (tempMax+tempMin)<0:
            tMin=tempMin+tempMax
            tMax=0
            minIndex=total.index(min(total))
            maxIndex=total.index(max(total))
            temp[minIndex]=tMin
            temp[maxIndex]=tMax
            step['from']=DictName[maxIndex]
            step['to']=DictName[minIndex]
            step['amount']=round(min(tempMax,abs(tempMin)),2)
            Trans.append(step)
            total=temp
        elif (tempMax+tempMin)==0 and tempMax==0 and tempMin==0:
            break;
        elif (tempMax+tempMin)==0 and tempMax!=0:
            tMax=0
            tMin=0
            minIndex=total.index(min(total))
            maxIndex=total.index(max(total))
            temp[minIndex]=tMin
            temp[maxIndex]=tMax
            step['from']=DictName[maxIndex]
            step['to']=DictName[minIndex]
            step['amount']=round(min(tempMax,abs(tempMin)),2)
            Trans.append(step)
            total=temp
    Out={'transactions':Trans}
    return jsonify(Out)

    