from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.joblib")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_input = np.array(features).reshape(1, -1)

    prediction = model.predict(final_input)

    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)