from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

# ✅ ترتيب الأعمدة كما تم أثناء التدريب
FEATURE_ORDER = [
    'Unnamed: 0', 'BMI', 'Smoking', 'AlcoholDrinking', 'Stroke',
    'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'AgeCategory',
    'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease',
    'SkinCancer', 'Race_Asian', 'Race_Black', 'Race_Hispanic', 'Race_Other',
    'Race_White', 'Diabetic_No, borderline diabetes', 'Diabetic_Yes',
    'Diabetic_Yes (during pregnancy)'
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # تأكد أن البيانات تحتوي كل الأعمدة بنفس الترتيب
        df = pd.DataFrame([data])[FEATURE_ORDER]

        prediction = model.predict(df)
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
