from flask import Flask,request
from datetime import datetime

app = Flask(__name__)


@app.route('/getSklearn',methods = ['POST'])
def getSklearnMetrics():
    return {"TimeStamp":1,"Accuracy":1}

@app.route('/getTorch',methods = ['POST'])
def getTorchMetrics():
    data = request.form
    return {"TimeStamp":1,"Accuracy":1}

@app.route('/getKeras',methods = ['POST'])
def getKerasMetrics():
    return {"TimeStamp":1,"Accuracy":1}