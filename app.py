import numpy as np
import math
from flask import Flask, request, render_template
import tensorflow
from tensorflow import keras

# Create app
app = Flask(__name__)
model = keras.models.load_model("model/model_sequelize.h5")


@app.route("/")
def Home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    int_features = [[int(x) for x in request.form.values()]]
    features = [np.array(int_features)]
    prediction = model.predict(features)

    output = {0: "Customers tend not to leave", 1: "Customers tend to leave"}

    return render_template("predict.html",
                           prediction_text="{}".format(output[(int(
                               math.ceil(prediction[0][0])))]))


@app.route('/data-set')
def dataSet():
    return render_template('telcodataset.html')


@app.route('/predict')
def predicts():
    return render_template('predict.html')


if __name__ == "__main__":
    app.run(debug=True)