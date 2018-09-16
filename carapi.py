from flask import Flask, request, jsonify
from sklearn.externals import joblib

app = Flask(__name__)

# Load the model
MODEL = joblib.load('car-DT-v1.0.pkl')
MODEL_LABELS = ['unacc', 'acc', 'vgood', 'good']

@app.route('/predict')
def predict():
    # Retrieve query parameters related to this request.
    Buying = request.args.get('Buying')
    Maint = request.args.get('Maint')
    doors = request.args.get('doors')
    lug_boot = request.args.get('lug_boot')
    safety = request.args.get('safety')

    # Our model expects a list of records
    features = [[Buying, Maint, doors, lug_boot, safety]]


    # predict the class and probability of the class
    label_index = MODEL.predict(features)
    label_conf = MODEL.predict_proba(features)

    # Retrieve the name of the predicted class
    label = MODEL_LABELS[label_index[0]]

    # Create and send a response to the API caller
    return jsonify(status='complete', label=label,  label_conf = ''.join(str(label_conf)))



if __name__ == '__main__':
    app.run(debug=True)
