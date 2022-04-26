from flask import Flask, render_template,url_for, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html", title="home page")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    input_features = [float(x) for x in request.form.values()]
    final_input = [np.array(input_features)]
    prediction = model.predict(final_input)
    output = prediction[0]  # getting first index
    if output == 1:
        return render_template('home.html', pred="You have very high chances of being diabetic in the future")
    else:
        return render_template("home.html", pred="You have very low chances of being diabetic in the future")


if __name__ == "__main__":
    app.run(debug=False)
