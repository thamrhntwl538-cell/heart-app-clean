from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("heart_app/model.pkl")
scaler = joblib.load("heart_app/scaler.pkl")

features = [
    "age","sex","cp","trestbps","chol","fbs",
    "restecg","thalach","exang","oldpeak",
    "slope","ca","thal"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = [float(request.form[f]) for f in features]

    df = pd.DataFrame([data], columns=features)
    scaled = scaler.transform(df)

    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0]

    if pred == 1:
        result = "⚠️ Heart disease detected"
    else:
        result = "✅ Healthy"

    # 🔥 Most important features
    importances = model.feature_importances_
    top_features = sorted(
        zip(features, importances),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    # 🧠 Convert data to dictionary
    patient = dict(zip(features, data))

    # 🧠 Smart analysis based on values
    analysis = []

    if patient["age"] > 55:
        analysis.append(f"👴 Age is high ({patient['age']}) which increases heart disease risk")

    if patient["chol"] > 240:
        analysis.append(f"🧈 High cholesterol ({patient['chol']}) is a risk factor")

    if patient["trestbps"] > 140:
        analysis.append(f"🩺 High blood pressure ({patient['trestbps']})")

    if patient["thalach"] < 100:
        analysis.append(f"❤️ Low heart rate ({patient['thalach']})")

    if patient["oldpeak"] > 2:
        analysis.append(f"⚡ Heart stress detected (oldpeak = {patient['oldpeak']})")

    if patient["exang"] == 1:
        analysis.append("⚠️ Exercise-induced chest pain detected")

    # If no issues
    if not analysis:
        analysis.append("✅ Values appear normal with no clear risk indicators")

    # 💡 Smart advice
    advice = []

    if pred == 1:
        if patient["chol"] > 240:
            advice.append("🥗 Reduce cholesterol in your diet")

        if patient["trestbps"] > 140:
            advice.append("🩺 Monitor blood pressure regularly")

        if patient["thalach"] < 100:
            advice.append("🏃‍♂️ Improve cardiovascular fitness")

        advice.append("🚶‍♂️ Exercise regularly")
        advice.append("🥦 Maintain a healthy diet")

    else:
        advice = [
            "✅ Keep up your healthy lifestyle",
            "🏃 Stay active",
            "🥗 Maintain a balanced diet"
        ]

    return render_template(
        "index.html",
        result=result,
        no_disease=f"{prob[0]*100:.2f}",
        disease=f"{prob[1]*100:.2f}",
        top_features=top_features,
        analysis=analysis,
        advice=advice
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
