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

def MinDiff(arr):
    n=len(arr)
    # Initialize difference as infinite

    arr = sorted(arr)
 
    # Initialize difference as infinite
    diff = 10**20
 
    # Find the min diff by comparing adjacent
    # pairs in sorted array
    for i in range(n-1):
        if arr[i+1] - arr[i] < diff:
            diff = arr[i+1] - arr[i]
 
    # Return min diff
    return diff



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
    result = {"answer" : answer}
    print ("FINAL ANSWER :  ", result)
    return jsonify(result)

    
@app.route('/tally-expense', methods=['POST'])
def getOut():
    Data = request.get_json();
    print(Data)
    Person=Data["persons"]
    n=len(Person)
    Expense=Data["expenses"]
    billN=len(Expense)
    namePay={}
    for i in range(n):
        namePay[Person[i]]=0
        DictName={}
    for p in range(n):
        DictName[p]=Data['persons'][p]
    ##
    def getShouldPay(exclude):
        L=[]
        for x in Person:
            if x not in exclude:L.append(x)
        return L
    ##
    for i in range(billN):
        AllShare=0
        now=Expense[i]
        namePay[now['paidBy']]-=now['amount']
        if 'exclude' in now.keys():
            ShouldPay=getShouldPay(now["exclude"])
            Share=now['amount']/len(ShouldPay)
            for k in range(len(ShouldPay)):
                namePay[ShouldPay[k]]+=Share
        if 'exclude' not in now.keys():
            AllShare+=now['amount']
    allShare=AllShare/n
    for j in range(n):
        namePay[Person[j]]+=allShare
        listPay=[]
    for d in range(n):
        listPay.append(namePay[Person[d]])
    ##
    total=listPay
    trans=[]
    while max(total)>0.05:
        temp=copy.copy(total)
        step={'from':0,'to':0,'amount':0}
        minIndex=total.index(min(total))
        maxIndex=total.index(max(total))
        tempMin=temp[minIndex]
        tempMax=temp[maxIndex]
        if(tempMax+tempMin)>0:
            tMax=tempMax+tempMin
            tMin=0
            temp[minIndex]=tMin
            temp[maxIndex]=tMax
            step['from']=DictName[maxIndex]
            step['to']=DictName[minIndex]
            step['amount']=round(min(tempMax,abs(tempMin)),2)
            total=temp
            trans.append(step)
        if(tempMax+tempMin)<0:
            tMax=0
            tMin=tempMax+tempMin
            temp[minIndex]=tMin
            temp[maxIndex]=tMax
            step['from']=DictName[maxIndex]
            step['to']=DictName[minIndex]
            step['amount']=min(tempMax,abs(tempMin))
            total=temp
            trans.append(step)
    Trans={'transactions':trans}
    return jsonify(Trans)

@app.route('/airtrafficcontroller', methods=['POST'])
def sortFlights():
    json_file = request.get_json();
    print(json_file)
    flights = json_file['Flights']
    times=[]
    for flight in flights:
        times.append(flight['Time'])
    times.sort()
    answer_flights=[]
    for time in times:
        for flight in flights:
            added=0
            if flight['Time']==time and added==0:
                print(time)
                answer_flights.append(flight)
                added=1

    answer = {"Flights" : answer_flights }
    print(answer)
    return jsonify(answer_flights)

@app.route('/customers-and-hotel/minimum-distance', methods=['POST'])
def mind():
    Data = request.get_json();
    print(Data)
    min_distance = MinDiff(Data)
    answer = {"answer" : min_distance}
    return jsonify(answer)

@app.route('/customers-and-hotel/minimum-camps', methods=['POST'])
def camps():
    data = request.get_json();
    print(data)
    answer = len(data)
    final_answer = {"answer" : answer}
    return jsonify(final_answer)

@app.route('/two-dinosaurs', methods=['POST'])
def CalResult():
    data = request.get_json();
    print(data)
    Q=data["maximum_difference_for_calories"]
    A=data["calories_for_each_type_for_raphael"]
    result=0
    N=data["number_of_types_of_food"]
    B=data["calories_for_each_type_for_leonardo"]
    NumOutCome=2**(N*2)
    food=np.array(A+B)
    for k in range(NumOutCome):
           kk=bin(k)[2:]
           if len(kk)<(N*2):kk='0'*(N*2-len(kk))+kk
           a=np.array((list(kk)),dtype=int)
           temp=copy.copy(food)
           temp[np.argwhere(a==0)]=0
           if abs(sum(temp[0:N])-sum(temp[N:]))<=Q:result+=1
    if result>2**15:result%=100000123
    final_result ={"result" : result}
    return jsonify(final_result)


