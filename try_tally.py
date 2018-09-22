
import json
import flask
from flask import request, jsonify
import copy
@app.route('/tally-expense', methods=['POST'])
def getOut():
    Data = request.get_json();
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
    temp=copy.copy(total)
    Trans=[]
    #temp=copy.copy(total)
    while total!=[0]*n:
        #print(total)
        temp=copy.copy(total)
        step={'from':0,'to':0,'amount':0}
        tempMin=temp[total.index(min(total))]
        tMin=tempMin
        tempMax=temp[total.index(max(total))]
        tMax=tempMax
        if(tempMax+tempMin)>0:
            tMax=tempMax+tempMin
            tMin=0
            temp[total.index(min(total))]=tMin
            temp[total.index(max(total))]=tMax
            step['from']=DictName[total.index(max(total))]
            step['to']=DictName[total.index(min(total))]
            step['amount']=round(min(abs(tempMax),abs(tempMin)),2)
            Trans.append(step)
            total=temp
        elif (tempMax+tempMin)<0:
            tMin=tempMin+tempMax
            tMax=0
            temp[total.index(min(total))]=tMin
            temp[total.index(max(total))]=tMax
            step['from']=DictName[total.index(max(total))]
            step['to']=DictName[total.index(min(total))]
            step['amount']=round(min(abs(tempMax),abs(tempMin)),2)
            Trans.append(step)
            total=temp
        elif (tempMax+tempMin)==0 and tempMax==0 and tempMin==0:
            break;
        elif (tempMax+tempMin)==0 and tempMax!=0:
            tMax=0
            tMin=0
            temp[total.index(min(total))]=tMin
            temp[total.index(max(total))]=tMax
            step['from']=DictName[total.index(max(total))]
            step['to']=DictName[total.index(min(total))]
            step['amount']=round(min(abs(tempMax),abs(tempMin)),2)
            Trans.append(step)
            total=temp
    Out={'transaction':Trans}
    return json.dumps(Out)
