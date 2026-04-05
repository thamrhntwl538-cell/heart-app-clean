from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

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

    result = "⚠️ لديه مرض قلب" if pred == 1 else "✅ سليم"

    patient = dict(zip(features, data))

    # تحليل
    analysis = []
    if patient["age"] > 55:
        analysis.append(f"👴 العمر مرتفع ({patient['age']})")
    if patient["chol"] > 240:
        analysis.append(f"🧈 الكوليسترول عالي ({patient['chol']})")
    if patient["trestbps"] > 140:
        analysis.append(f"🩺 ضغط الدم مرتفع ({patient['trestbps']})")
    if patient["thalach"] < 100:
        analysis.append(f"❤️ نبض القلب منخفض ({patient['thalach']})")
    if not analysis:
        analysis.append("✅ القيم طبيعية")

    # مقارنة
    comparison = []
    comparison.append(f"Cholesterol: {patient['chol']}")
    comparison.append(f"Blood Pressure: {patient['trestbps']}")
    comparison.append(f"Heart Rate: {patient['thalach']}")

    # نصائح
    advice = ["🏃‍♂️ مارس الرياضة", "🥗 غذاء صحي"]

    # تقرير
    report = f"""
Result: {result}

Disease: {prob[1]*100:.2f}%
"""

    return render_template(
        "index.html",
        result=result,
        no_disease=f"{prob[0]*100:.2f}",
        disease=f"{prob[1]*100:.2f}",
        analysis=analysis,
        comparison=comparison,
        advice=advice,
        report=report
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
