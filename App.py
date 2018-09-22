import logging
from codeitsuisse import  app
import itertools
import urllib.request
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import json

#Imports for Deep learning
import copy
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import load_model
import numpy
#Imports for Deep Learning trade here

from flask import request, jsonify;


logger = logging.getLogger(__name__)

def get_primes(n):
    numbers = set(range(n, 1, -1))
    primes = []
    while numbers:
        p = numbers.pop()
        primes.append(p)
        numbers.difference_update(set(range(p*2, n+1, p)))
    return primes




@app.route('/', methods=['GET'])
def default_route():
    return "Team typhoont10 page";

if __name__ == "__main__":
    logFormatter = logging.Formatter("%(asctime)s [%(filename)s] [%(funcName)s] [%(lineno)d] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()

    rootLogger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler("team.log")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    logger.info("Starting application ...")
    app.run()

@app.route('/hello', methods=['GET'])
def say_hello():
	return "HELLO, how are you?";


answer_list={}

@app.route('/prime-sum', methods=['POST'])
def prime_sum():
    data = request.get_json();
    logging.info("data sent for evaluation {}".format(data))
    N = data.get("input");
    N=int(N)
    print(N)
    prime_list = get_primes(N)
    answer=[]
    #Loop to all the possible number of prime numbers
    for L in range(0, len(prime_list)+1):
        for subset in itertools.combinations(prime_list, L):
            if (sum(subset)==N):
                answer = subset
    return str(list(answer))

@app.route('/imagesGPS', methods=['POST'])        
def getGPS():
    answer=[]
    data = request.get_json()

    print(data)
    for path_dict in data:
        url=path_dict['path']
        urllib.request.urlretrieve(url,'abc.jpg') 
        def _convert_to_degress(value):
            d0 = value[0][0]
            d1 = value[0][1]
            d = float(d0) / float(d1)
            m0 = value[1][0]
            m1 = value[1][1]
            m = float(m0) / float(m1)
            s0 = value[2][0]
            s1 = value[2][1]
            s = float(s0) / float(s1)
            return d + (m / 60.0) + (s / 3600.0)
        img= Image.open('abc.jpg')
        exifinfo=img._getexif()
        ret={}
        if exifinfo != None:
            for tag, value in exifinfo.items():
                decoded = TAGS.get(tag, tag)
                ret[decoded] = value
        #Latitude
        Lat=ret['GPSInfo'][2]
        #longitude
        Long=ret['GPSInfo'][4]
        lat=_convert_to_degress(Lat)
        long=_convert_to_degress(Long)
        out={'lat':lat,'long':long}
        answer.append(out)
    return jsonify(answer)

@app.route('/machine-learning/question-1', methods=['POST'])
def deepLearning():
    data = request.get_json();
    print(data)
    X=np.array(data['input'])
    Y=np.array(data['output'])
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
    print("ANSWER == ", answer)
    ans_dict= {"answer" : answer}
    print("ANSER DICTIONARY  = ", ans_dict)
    return jsonify(ans_dict)
        
@app.route('/machine-learning/question-2', methods=['POST'])
def deepLearning2():
    model = load_model('models/mnistCNN.h5')
    answer=[]
    data = request.get_json()
    images = data['question']
    print(images)
    for image in images:
        image = np.array(image)
        image = image.reshape(28,28)
        im2arr = image.reshape(1,28,28,1)
        y_pred = model.predict(im2arr)
        
        y_pred=y_pred[0]
        y_pred = y_pred.tolist()
        result = y_pred.index(1.0)
        answer.append(result)
        print("IMAGE 1 : ", answer)
    answer=str(answer)
    result = {"answer" : answer}
    print ("FINAL ANSWER :  ", result)
    return jsonify(result)

    
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
    return json.dumps(Out)

    