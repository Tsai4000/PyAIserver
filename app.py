import pymongo
import sys
sys.path.insert(1, './AIutil/kannada')

from pyAI import *
from flask import Flask, jsonify, request, render_template, make_response

app = Flask(__name__)

def initDB():
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")

    dblist = myclient.list_database_names()
    if "pyAI" in dblist:
        print("已存在！")
        mydb = myclient['pyAI']
        collist = mydb.list_collection_names()
        if "NN" in collist:  
            print("集合已存在！")
    print("mongo init success")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def bPredict():
    # print(request.get_json(), file=sys.stderr)
    result, perc = predict(request.get_json()["icon"][22:])
    return make_response(jsonify({"result": str(result), "perc": str(perc)}), 200)

initDB()
app.run(host='0.0.0.0', port=5000, debug=True)