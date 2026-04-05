from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# تحميل الموديل
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
        result = "⚠️ لديه مرض قلب"
    else:
        result = "✅ سليم"

    # أهم العوامل
    importances = model.feature_importances_
    top_features = sorted(
        zip(features, importances),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    patient = dict(zip(features, data))

    # 🧠 تحليل ذكي
    analysis = []

    if patient["age"] > 55:
        analysis.append(f"👴 العمر مرتفع ({patient['age']})")

    if patient["chol"] > 240:
        analysis.append(f"🧈 الكوليسترول عالي ({patient['chol']})")

    if patient["trestbps"] > 140:
        analysis.append(f"🩺 ضغط الدم مرتفع ({patient['trestbps']})")

    if patient["thalach"] < 100:
        analysis.append(f"❤️ نبض القلب منخفض ({patient['thalach']})")

    if patient["oldpeak"] > 2:
        analysis.append(f"⚡ إجهاد على القلب ({patient['oldpeak']})")

    if patient["exang"] == 1:
        analysis.append("⚠️ ألم صدر أثناء التمرين")

    if not analysis:
        analysis.append("✅ القيم طبيعية")

    # 📊 مقارنة بالقيم الطبيعية
    comparison = []

    if patient["chol"] > 240:
        comparison.append(f"Cholesterol: {patient['chol']} (High ❗ | <200)")
    else:
        comparison.append(f"Cholesterol: {patient['chol']} (Normal ✅)")

    if patient["trestbps"] > 140:
        comparison.append(f"Blood Pressure: {patient['trestbps']} (High ❗ | ~120)")
    else:
        comparison.append(f"Blood Pressure: {patient['trestbps']} (Normal ✅)")

    if patient["thalach"] < 100:
        comparison.append(f"Heart Rate: {patient['thalach']} (Low ⚠️)")
    else:
        comparison.append(f"Heart Rate: {patient['thalach']} (Good ✅)")

    if patient["fbs"] == 1:
        comparison.append("Blood Sugar: High ❗")
    else:
        comparison.append("Blood Sugar: Normal ✅")

    # 💡 توصيات
    advice = []

    if pred == 1:
        advice.append("🥗 قلل الدهون")
        advice.append("🏃‍♂️ مارس الرياضة")
        advice.append("🩺 راقب ضغط الدم")
    else:
        advice.append("✅ استمر على نمطك الصحي")
        advice.append("🏃 حافظ على نشاطك")

    # 🧾 تقرير كامل
    report = f"""
Patient Report

Result: {result}

Risk:
- Disease: {prob[1]*100:.2f}%
- No Disease: {prob[0]*100:.2f}%

Analysis:
"""
    for a in analysis:
        report += f"- {a}\n"

    report += "\nComparison:\n"
    for c in comparison:
        report += f"- {c}\n"

    report += "\nAdvice:\n"
    for a in advice:
        report += f"- {a}\n"

    return render_template(
        "index.html",
        result=result,
        no_disease=f"{prob[0]*100:.2f}",
        disease=f"{prob[1]*100:.2f}",
        top_features=top_features,
        analysis=analysis,
        comparison=comparison,
        advice=advice,
        report=report
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
