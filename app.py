import pymongo
import sys
sys.path.insert(1, './AIutil')

from pyAI import *
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

def initDB():
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")

    dblist = myclient.list_database_names()
    if "pyAI" in dblist:
        print("已存在！")
        mydb = myclient['pyAI']
        collist = mydb.list_collection_names()
        if "NN" in collist:   # 判断 sites 集合是否存在
            print("集合已存在！")
    print("mongo init success")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def bPredict():
    # print(request.form["icon"][22:], file=sys.stderr)
    result = predict(request.form["icon"][22:])
    return jsonify({"result": str(result)})

initDB()
app.run(port=5000, debug=True)