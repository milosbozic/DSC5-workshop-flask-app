import pandas as pd
from flask import Flask, jsonify, request
import pickle
import bz2

# load model
#model = pickle.load(open('final_model.pkl','rb'))
with bz2.open('final_model', 'r') as fp:
    loaded_model = pickle.load(fp)

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

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