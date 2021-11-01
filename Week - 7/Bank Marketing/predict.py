
import pickle
from flask import Flask
from flask import request
from flask import jsonify
import xgboost as xgb

model_file = 'final_model=1.0.bin'

# loading the dict-vectorizer and model
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)
print("DictVectoriser and Model are loaded.")

app = Flask('subscribe')

# home page
@app.route('/home', methods=['GET'])
def home_page():
    message = "Welcome to home page of Flask App, deployed as docker image."
    return message

# prediction page
@app.route('/predict', methods=['POST'])
def predict():
    print("Requesting input data(customer) as json.........")
    customer = request.get_json()
    print("input data(customer) received ..................")
    X = dv.transform([customer])
    features = dv.get_feature_names()
    dcust = xgb.DMatrix(X, feature_names=features)
    print('Predicting Probability for term depoist........')
    y_pred = model.predict(dcust)[0]
    subs = y_pred >= 0.7
    print('Subscribe probability: ', (y_pred*100))

    result = {
        "subscription-probablity": float(y_pred),
        "subscribe": bool(subs)
    }
    print('Prediction result sent! .......................')
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9697)

