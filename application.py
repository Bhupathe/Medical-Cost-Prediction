import pickle
from flask import Flask, render_template, jsonify, request
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('models/poly_reg_2.pkl', 'rb'))
poly_features = pickle.load(open('models/poly_features.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            age = int(request.form["age"])
            sex = request.form["sex"]
            bmi = float(request.form["bmi"])
            children = int(request.form["children"])
            smoker = request.form["smoker"]
            region = request.form["region"]

            # Validate categorical values
            if sex not in ["male", "female"]:
                raise ValueError("Please select a valid gender.")

            if smoker not in ["yes", "no"]:
                raise ValueError("Please select a valid smoking status.")

            if region not in [
                "northeast", "northwest", "southeast", "southwest"
            ]:
                raise ValueError("Please select a valid region.")

            # Match the encoding used during model training
            sex_encoded = 1 if sex == "female" else 0
            smoker_encoded = 1 if smoker == "yes" else 0

            # Build a feature row with the same columns and order
            # as the data used to train the model.
            input_data = pd.DataFrame([{
                "age": age,
                "sex": sex_encoded,
                "bmi": bmi,
                "children": children,
                "smoker": smoker_encoded,
                "northeast": int(region == "northeast"),
                "northwest": int(region == "northwest"),
                "southeast": int(region == "southeast"),
            }])

            input_data = poly_features.transform(input_data)
            prediction = float(model.predict(input_data)[0])

        except (ValueError, KeyError) as e:
            error = str(e) or "Please enter valid values."

    return render_template(
        "home.html",
        prediction=prediction,
        error=error
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5020)