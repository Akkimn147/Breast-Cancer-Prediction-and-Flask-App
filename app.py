import numpy as np
from flask import Flask, request, json, render_template
import pickle

#Create Flask app
app =Flask(__name__)

#Load the pickle model

model = pickle.load(open("trainedmodel.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods= ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "Breast cancer detected Yes (1) or No (0):- {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)