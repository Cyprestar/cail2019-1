from flask import Flask, request, jsonify
from flask_cors import *
# from model import MatchModel
import json
import random

app = Flask(__name__)
CORS(app, resources={r'/*'}, supports_credentials=True)

# MODEL_DIR = '../output/model'
# model = MatchModel.load(MODEL_DIR)

texts = []
with open('../data/raw/CAIL2019-SCM-big/input.json', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        d = json.loads(line.strip())
        if 'label' in d:
            del (d['label'])
        texts.append(d)


@app.route('/predict', methods=['POST'])
def predict():
    obj = request.get_json()
    a, b, c = obj['A'], obj['B'], obj['C']
    # results = model.predict([(a, b, c)])
    # label, _ = results[0]
    # return jsonify({'result': label})
    return jsonify({'data': [a, b, c]})


@app.route('/random', methods=['GET'])
def get_random():
    idx = random.randrange(len(texts))
    return jsonify(texts[idx])


if __name__ == '__main__':
    app.run()
