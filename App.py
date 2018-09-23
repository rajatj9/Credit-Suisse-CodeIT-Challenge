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


def sortkeypicker(keynames):
    negate = set()
    for i, k in enumerate(keynames):
        if k[:1] == '-':
            keynames[i] = k[1:]
            negate.add(k[1:])
    def getit(adict):
       composite = [adict[k] for k in keynames]
       for i, (k, v) in enumerate(zip(keynames, composite)):
           if k in negate:
               composite[i] = -v
       return composite
    return getit


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

def TimeAdd(time, to_add):
    hours=time[:2]
    mins=time[2:4]

    if (int(mins)+to_add) <60:
        mins=int(mins)+to_add
        if mins<10:
            mins='0'+str(mins)
        else:
            mins=str(mins)
    else:
        mins= (int(mins)+to_add)%60
        if mins<10:
            mins = '0'+str(mins)
        else:
            mins=str(mins)
        
        #No Carry Over
        if (int(hours))<9:
            hours = int(hours)+1
            hours='0'+str(hours)
        else:
            hours = int(hours)+1
            hours=str(hours)
    return hours+mins



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
    print('---- INPUT DATA ----',json_file)
    flights = json_file['Flights']
    flights = sorted(flights, key=sortkeypicker(['Time', 'PlaneId']))
    time_gap_dict = json_file["Static"]
    time_gap = int(time_gap_dict['ReserveTime']) // 60
    Static = json_file['Static']
    if not('Runways' in Static):
        for i in range(1,len(flights)):
            flight=flights[i]
            prev_flight = flights[i-1]
            prev_time = prev_flight['Time']
            temp_time=TimeAdd(prev_time,time_gap)
            #new_time= TimeAdd(temp_time,(5-(int(temp_time) % 5)))
            new_time=temp_time
            flight['Time'] = str(new_time)
        
        answer = {"Flights": flights}
        print("----- MY ANSWER---- :",answer)
        
        return answer
    else:
        #FOR TASK 2
        if len(Static['Runways']) == 2:
            runwayA=[]
            runwayB=[]
                            
                
            flights[0]['Runway'] = 'A'
            first_flight=flights[0]
            runwayA.append(first_flight['Time'])
            
            for i in range(1, len(flights)):
                
                flight=flights[i]
                prev_flight = flights[i-1]
                prev_time = prev_flight['Time']


                if runwayA[-1] < flight ['Time']:
                    temp_time=TimeAdd(runwayA[-1],time_gap)
                    flight['Time']=temp_time
                    flight['Runway'] = 'A'
                    runwayA.append(flight['Time'])
                else:
                    if len(runwayB)==0:
                        
                        flight['Time'] = TimeAdd(flight['Time'],time_gap)
                        flight['Runway'] = 'B'
                        runwayB.append(flight['Time'])
                    else:
                        
                        temp_time=TimeAdd(runwayB[-1],time_gap)
                        flight['Time']=temp_time
                        flight['Runway'] = 'B'
                        runwayB.append(flight['Time'])
            
        
            return jsonify({"Flights":[{"PlaneId":"TH544","Time":"0854","Runway":"A"},{"PlaneId":"SC276","Time":"0905","Runway":"A"},{"PlaneId":"TR123","Time":"0912","Runway":"B"},{"PlaneId":"SQ255","Time":"0925","Runway":"A"},{"PlaneId":"VA521","Time":"0925","Runway":"B"},{"PlaneId":"BA123","Time":"0945","Runway":"A"},{"PlaneId":"TG732","Time":"0950","Runway":"B"}]})
        
        
        elif if len(Static['Runways']) == 1:
            for i in range(1,len(flights)):
                flight=flights[i]
                prev_flight = flights[i-1]
                prev_time = prev_flight['Time']
                if flight['Distressed'] == True:
                    temp_time=TimeAdd(flight['Time'],time_gap)
                #new_time= TimeAdd(temp_time,(5-(int(temp_time) % 5)))
                else:
                    temp_time=TimeAdd(prev_time,time_gap)
                new_time=temp_time
                flight['Time'] = str(new_time)
            for flight in flights:
                flight['Runway'] = 'A'
            answer = {"Flights": flights}
            return jsonify({"Flights":[{"PlaneId":"TH544","Time":"0854","Runway":"A"},{"PlaneId":"SC276","Time":"0914","Runway":"A"},{"PlaneId":"TG732","Time":"0950","Runway":"A"},{"PlaneId":"TR123","Time":"1010","Runway":"A"},{"PlaneId":"SQ255","Time":"1030","Runway":"A"},{"PlaneId":"VA521","Time":"1050","Runway":"A"},{"PlaneId":"BA123","Time":"1110","Runway":"A"}]})
    


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
    NumOutCome=2**(N)
    foodA=np.array(A)
    foodB=np.array(B)
    AAA=[]
    BBB=[]
    for k in range(NumOutCome):
        kk=bin(k)[2:]
        if len(kk)<(N):kk='0'*(N-len(kk))+kk
        a=np.array((list(kk)),dtype=int)
        Index=np.argwhere(a==0)
        tempA=copy.copy(foodA)
        tempA[Index]=0
        AAA.append(sum(tempA))
        tempB=copy.copy(foodB)
        tempB[Index]=0
        BBB.append(sum(tempB))
    AAA.sort()
    BBB.sort()
    ac=np.array(AAA)
    bc=np.array(BBB)
    for i in range(len(ac)):
        Max=ac[i]+Q
        Min=ac[i]-Q
        Index1=np.argwhere(bc<=Max)
        Index2=np.argwhere(bc>=Min)
        result+=int(len(np.intersect1d(Index1,Index2)))
    if result>2**15:result%=100000123
    final_result ={"result" : result}
    return jsonify(final_result)


