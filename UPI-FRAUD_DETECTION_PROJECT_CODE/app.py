from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model, scaler, and encoder
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get inputs
        trans_hour = int(request.form["trans_hour"])
        trans_day = int(request.form["trans_day"])
        trans_month = int(request.form["trans_month"])
        trans_year = int(request.form["trans_year"])
        trans_amount = float(request.form["trans_amount"])
        upi_number = str(request.form["upi_number"])

        # Encode UPI number — handle unseen ones
        if upi_number in label_encoder.classes_:
            upi_number_encoded = label_encoder.transform([upi_number])[0]
        else:
            # If unseen, assign a special value (like -1)
            upi_number_encoded = -1

        # Prepare features
        features = np.array([[trans_hour, trans_day, trans_month, trans_year, trans_amount, upi_number_encoded]])
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]
        result = "Fraud" if prediction == 1 else "Not Fraud"

        return render_template("index.html", result=result)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
