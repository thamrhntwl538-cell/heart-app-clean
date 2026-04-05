from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# تحميل الموديل
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

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

    return render_template(
        "index.html",
        result=result,
        no_disease=f"{prob[0]*100:.2f}",
        disease=f"{prob[1]*100:.2f}",
        top_features=top_features
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)