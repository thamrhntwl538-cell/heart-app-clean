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

    # 🔥 أهم العوامل
    importances = model.feature_importances_
    top_features = sorted(
        zip(features, importances),
        key=lambda x: x[1],
        reverse=True
    )[:3]

    # 🧠 تحويل البيانات لقاموس
    patient = dict(zip(features, data))

    # 🧠 تحليل ذكي حسب القيم
    analysis = []

    if patient["age"] > 55:
        analysis.append(f"👴 العمر مرتفع ({patient['age']}) يزيد خطر أمراض القلب")

    if patient["chol"] > 240:
        analysis.append(f"🧈 الكوليسترول عالي ({patient['chol']}) وهذا عامل خطر")

    if patient["trestbps"] > 140:
        analysis.append(f"🩺 ضغط الدم مرتفع ({patient['trestbps']})")

    if patient["thalach"] < 100:
        analysis.append(f"❤️ نبض القلب منخفض ({patient['thalach']})")

    if patient["oldpeak"] > 2:
        analysis.append(f"⚡ يوجد إجهاد على القلب (oldpeak = {patient['oldpeak']})")

    if patient["exang"] == 1:
        analysis.append("⚠️ يوجد ألم صدر أثناء التمرين")

    # لو ما فيه مشاكل
    if not analysis:
        analysis.append("✅ القيم تبدو طبيعية ولا تشير لخطر واضح")

    # 💡 توصيات ذكية
    advice = []

    if pred == 1:
        if patient["chol"] > 240:
            advice.append("🥗 قلل الكوليسترول في الأكل")

        if patient["trestbps"] > 140:
            advice.append("🩺 راقب ضغط الدم باستمرار")

        if patient["thalach"] < 100:
            advice.append("🏃‍♂️ حاول تحسين اللياقة القلبية")

        advice.append("🚶‍♂️ مارس الرياضة بانتظام")
        advice.append("🥦 تناول غذاء صحي")

    else:
        advice = [
            "✅ استمر على نمطك الصحي",
            "🏃 حافظ على نشاطك",
            "🥗 استمر في الغذاء المتوازن"
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
