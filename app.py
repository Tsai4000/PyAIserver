import pymongo
import sys
import os
import time
sys.path.insert(1, './AIutil/kannada')
sys.path.insert(1, './AIutil/snake')

from pyAI import *
from snakeAI import *
from flask import Flask, jsonify, request, render_template, make_response

app = Flask(__name__)
myclient = None
geneWeight = None
geneLog = None

playerQ = 0
resPlayerQ = 0
POP_MAX = 50
playerQList = [False for i in range(POP_MAX)]
def initDB():
    global myclient, geneWeight, geneLog
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    geneWeight = myclient['dbtSnake']['geneWeight']
    geneLog = myclient['dbtSnake']['geneLog']
    if None in [myclient, geneWeight, geneLog]:
        print("monogo Failed")
        os._exit(0)
    print("mongo init success")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/snake')
def snake():
    global playerQ
    return render_template('snake.html')

@app.route('/predict', methods=['POST'])
def bPredict():
    # print(request.get_json(), file=sys.stderr)
    result, perc = predict(request.get_json()["icon"][22:])
    return make_response(jsonify({"result": str(result), "perc": str(perc)}), 200)

@app.route('/snake/gene', methods=['GET'])
def gGene():
    global myclient, playerQ, resPlayerQ, playerQList, geneWeight, geneLog
    if(playerQ < POP_MAX):
        weight = geneWeight.find({"weightIndex": playerQ})
        generation = geneLog.find().sort([("_id",-1)]).limit(1)
        playerQ+=1
        print(playerQ,resPlayerQ, file=sys.stderr)

        return make_response(jsonify({"index": playerQ-1, "generation": generation[0]['generation'], "weight": reshapeMat(weight[0]['weight'])}), 200)

    elif(playerQ == POP_MAX and resPlayerQ == POP_MAX):
        playerQ+=1
        weights=[]
        allWeight = geneWeight.find().sort('score')
        generation = geneLog.find().sort([("_id",-1)]).limit(1)[0]
        for weight in allWeight:
            weights.append(weight['weight'])
        newGene = calNextGene(weights)
        geneLog = {
            "generation": generation['generation']+1,
            'weight': weights
        }
        geneLog.insert_one(geneLog)
        geneWeight.delete_many({})
        for index, gene in enumerate(newGene):
            weight = {
                "weightIndex": index,
                "weight": gene.astype(float).tolist(),
                "score": -5000
            }
            geneWeight.insert_one(weight)
        playerQ = 1
        resPlayerQ = 0
        playerQList = [False for i in range(POP_MAX)]
        return make_response(jsonify({"index": 0, "generation": generation["generation"], "weight": reshapeMat(newGene[0])}), 200)
    elif(playerQ == POP_MAX and resPlayerQ < POP_MAX):
        unfinish = [index for index, i in enumerate(playerQList) if i is False][0]
        weight = geneWeight.find({"weightIndex": unfinish})
        generation = geneLog.find().sort([("_id",-1)]).limit(1)
        # playerQ+=1
        print(playerQ,resPlayerQ, file=sys.stderr)

        return make_response(jsonify({"index": unfinish, "generation": generation[0]['generation'], "weight": reshapeMat(weight[0]['weight'])}), 200)

    else:
        return make_response(jsonify({"index": -1}), 200)

@app.route('/snake/gene', methods=['POST'])
def pGene():
    global myclient, playerQ, resPlayerQ, playerQList, geneWeight, geneLog
    req = request.get_json()
    generation = geneLog.find().sort([("_id",-1)]).limit(1)[0]
    if(req['generation'] == generation['generation']):
        weight = geneWeight.update_one({"weightIndex": req['index']}, {"$set": {"score": req['score']}})
        playerQList[req['index']] = True
        if(resPlayerQ!=POP_MAX):
            resPlayerQ+=1
        return make_response(jsonify({"result": "success, plz wait a moment and start next pop"}), 200)
    else:
        return make_response(jsonify({"index": -1}), 200)

# @app.route('/createLog', methods=['GET'])
# def sss():
#     global myclient
#     geneLog.insert_one({"generation":0})

# @app.route('/create', methods=['GET'])
# def ggg():
#     global myclient
#     sol_per_pop = 50
#     num_weights = 9*12+12*16+16*3

#     # Defining the population size.
#     pop_size = (sol_per_pop,num_weights)
#     #Creating the initial population.
#     new_population = np.random.choice(np.arange(-1,1,step=0.01),size=pop_size,replace=True)
#     for index, pop in enumerate(new_population):
#         ge = {
#             "weightIndex": index,
#             "weight": pop.astype(float).tolist(),
#             "score": -5000
#         }
#         geneWeight.insert_one(ge)


initDB()
app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)