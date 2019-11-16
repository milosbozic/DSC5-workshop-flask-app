import pandas as pd
from flask import Flask, jsonify, request, render_template, url_for
import pickle
import bz2
import json
import requests

# load model
#model = pickle.load(open('final_model.pkl','rb'))
with bz2.open('final_model', 'r') as fp:
    model = pickle.load(fp)

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['GET', 'POST'])
def index():

    result = None
    if request.method == 'POST':
        cn = request.form.get('cn')
        gr = request.form.get('gr')
        zden = request.form.get('zden')
        rt = request.form.get('rt')

        data = {"CN": cn, "GR": gr, "ZDEN": zden, "RT": rt}
        data.update((x, [y]) for x, y in data.items())
        result = model.predict(pd.DataFrame.from_dict(data))
    
    return render_template('view.html', result=result)

@app.route('/api', methods=['POST'])
def predict():
    # get data
    data = request.get_json(force=True)
    #print(data)
    
    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)

    # predictions
    result = model.predict(data_df)

    # send back to browser
    output = {'results': int(result[0])}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)