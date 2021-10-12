from flask import Flask
from datetime import datetime

app = Flask(__name__)


@app.route('/getSklearn')
def hello_world():
    return {"TimeStamp":1,"Accuracy":1}

@app.route('/getTorch')
def hello_world():
    return {"TimeStamp":1,"Accuracy":1}

@app.route('/getKeras')
def hello_world():
    return {"TimeStamp":1,"Accuracy":1}