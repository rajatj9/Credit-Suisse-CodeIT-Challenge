import logging
from codeitsuisse import  app
import itertools
import urllib.request
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import json


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

def is_prime(a):
    x = True 
    for i in (2, a):
            while x:
               if a%i == 0:
                   x = False
               else:
                   x = True


    if x:
        print True
    else:
        print False


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
    if is_prime(N):
        return '[{}]'.format(N)
        return "[0]"
    for L in range(0, len(prime_list)+1):
        for subset in itertools.combinations(prime_list, L):
            if (sum(subset)==N):
                answer = subset
    return str(list(answer))

@app.route('/imageGPS', methods=['POST'])        
def getGPS(url):
    answer=[]
    data = request.get_json()
    data=json.loads(data)
    logging.info("data sent for evaluation {}".format(data))
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
