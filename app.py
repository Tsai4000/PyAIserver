import pymongo
import sys
import time
sys.path.insert(1, './AIutil/kannada')
sys.path.insert(1, './AIutil/snake')

from pyAI import *
from snakeAI import *
from flask import Flask, jsonify, request, render_template, make_response

app = Flask(__name__)
myclient = None
playerQ = 0
resPlayerQ = 0
POP_MAX = 50
def initDB():
    global myclient
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

@app.route('/snake')
def snake():
    global playerQ
    # playerQ+=1
    return render_template('snake.html')

@app.route('/predict', methods=['POST'])
def bPredict():
    # print(request.get_json(), file=sys.stderr)
    result, perc = predict(request.get_json()["icon"][22:])
    return make_response(jsonify({"result": str(result), "perc": str(perc)}), 200)

@app.route('/snake/gene', methods=['GET'])
def gGene():
    global myclient, playerQ, resPlayerQ
    print(playerQ,resPlayerQ, file=sys.stderr)

    if(playerQ < POP_MAX):
        weight = myclient['dbtSnake']['geneWeight'].find({"weightIndex": playerQ})
        generation = myclient['dbtSnake']['geneLog'].find().sort([("_id",-1)]).limit(1)
        playerQ+=1
        # print(reshapeMat(weight[0]['weight']), file=sys.stderr)

        return make_response(jsonify({"index": playerQ, "generation": generation[0]['generation'], "weight": reshapeMat(weight[0]['weight'])}), 200)

    elif(playerQ == POP_MAX and resPlayerQ == POP_MAX):
        weights=[]
        allWeight = myclient['dbtSnake']['geneWeight'].find().sort('score')
        generation = myclient['dbtSnake']['geneLog'].find().sort([("_id",-1)]).limit(1)[0]
        for weight in allWeight:
            weights.append(weight['weight'])
        newGene = calNextGene(weights)
        geneLog = {
            "generation": generation['generation']+1,
            'weight': weights
        }
        myclient['dbtSnake']['geneLog'].insert_one(geneLog)
        myclient['dbtSnake']['geneWeight'].delete_many({})
        for index, gene in enumerate(newGene):
            weight = {
                "weightIndex": index,
                "weight": gene.astype(float).tolist(),
                "score": -5000
            }
            myclient['dbtSnake']['geneWeight'].insert_one(weight)
        playerQ = 1
        resPlayerQ = 0
        return make_response(jsonify({"index": 0, "generation": generation["generation"], "weight": reshapeMat(newGene[0])}), 200)
    else:
        return make_response(jsonify({"index": -1}), 200)

@app.route('/snake/gene', methods=['POST'])
def pGene():
    global myclient, playerQ, resPlayerQ
    req = request.get_json()
    generation = myclient['dbtSnake']['geneLog'].find().sort([("_id",-1)]).limit(1)[0]
    if(req['generation'] == generation['generation']):
        weight = myclient['dbtSnake']['geneWeight'].update_one({"weightIndex": req['index']}, {"$set": {"score": req['score']}})
        resPlayerQ+=1
        return make_response(jsonify({"result": "success, plz wait a moment and start next pop"}), 200)
    else:
        return make_response(jsonify({"index": -1}), 200)

# @app.route('/createLog', methods=['GET'])
# def sss():
#     global myclient
#     myclient['dbtSnake']['geneLog'].insert_one({"generation":0})

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
#         myclient['dbtSnake']['geneWeight'].insert_one(ge)


initDB()
app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)