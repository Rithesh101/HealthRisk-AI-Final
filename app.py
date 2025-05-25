from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np
import shap

app = Flask(__name__)

# Load models
diabetes_model = joblib.load('models/knn_model.pkl')
heart_model = joblib.load('models/heart_model.pkl')
cancer_model = joblib.load('models/bcancer_model.pkl')

scaler = joblib.load("models/scalerHeart.pkl")
label_encoder = joblib.load("models/bcancer_labl.pkl")
scaler_diabetes = joblib.load("models/scalerDiabetes.pkl")

disease_links = {
    "Diabetes": {
        "url": "https://www.cdc.gov/diabetes/index.html",
        "label": "CDC Diabetes Prevention Page"
    },
    "Heart Disease": {
        "url": "https://www.cdc.gov/heart-disease/prevention/index.html",
        "label": "CDC Heart Disease Prevention Page"
    },
    "Breast Cancer Disease": {
        "url": "https://www.cdc.gov/breast-cancer/prevention/index.html",
        "label": "CDC Breast Cancer Prevention Page"
    }
}

# with open("models/diabetes.pkl", "rb") as f:
#     diabetes_model = pickle.load(f)

# with open("models/heart.pkl", "rb") as f:
#     heart_model = pickle.load(f)

# with open("models/cancer.pkl", "rb") as f:
#     cancer_model = pickle.load(f)

# # Load SHAP explainers
# explainer_diabetes = joblib.load('models/diabetes_explainer.pkl')
# explainer_heart = joblib.load('models/heart_explainer.pkl')
# explainer_cancer = joblib.load('models/cancer_explainer.pkl')
# with open("models/explainer_diabetes.pkl", "rb") as f:
#     explainer_diabetes = pickle.load(f)

# with open("models/explainer_heart.pkl", "rb") as f:
#     explainer_heart = pickle.load(f)

# with open("models/explainer_cancer.pkl", "rb") as f:
#     explainer_cancer = pickle.load(f)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/diabetes", methods=["GET", "POST"])
def diabetes():
    if request.method == "POST":
        data = [float(request.form[field]) for field in request.form]
        input_array = np.array([data])
        prediction = diabetes_model.predict(input_array)[0]
        #explanation = explainer_diabetes(input_array)
         #shap_values=explanation.values[0].tolist()
         # Personalized lifestyle advice for diabetes
        # # 1. Collect data in the same order used during training
        # data = [float(request.form[field]) for field in request.form]

        # # 2. Convert to numpy array and reshape
        # input_array = np.array([data])

        # # 3. Apply scaler
        # input_scaled = scaler.transform(input_array)

        # # 4. Make prediction
        # prediction = diabetes_model.predict(input_scaled)[0]
        lifestyle_advice = []
        if prediction == 1:
            lifestyle_advice.extend([
                "Follow a balanced diet rich in whole grains, fruits, vegetables, and lean proteins.",
                "Limit sugar, refined carbs, and processed foods.",
                "Engage in regular physical activity like walking, cycling, or swimming.",
                "Maintain a healthy weight and aim for gradual weight loss if overweight.",
                "Monitor blood glucose levels regularly.",
                "Get regular health check-ups and stay consistent with prescribed medications.",
            ])
        else:
            lifestyle_advice.append("Maintain your healthy lifestyle to continue minimizing diabetes risk.")

        return render_template("result.html", disease="Diabetes", prediction=prediction,features=data, lifestyle_advice=lifestyle_advice, read_more_info=disease_links["Diabetes"])
    return render_template("diabetes.html")


@app.route("/heart", methods=["GET", "POST"])
def heart():
    if request.method == "POST":
        # 1. Collect data in the same order used during training
        data = [float(request.form[field]) for field in request.form]

        # 2. Convert to numpy array and reshape
        input_array = np.array([data])

        # 3. Apply scaler
        input_scaled = scaler.transform(input_array)

        # 4. Make prediction
        prediction = heart_model.predict(input_scaled)[0]
       
        lifestyle_advice = []
        if prediction == 1:  # At risk
            lifestyle_advice.extend([
                "Engage in at least 30 minutes of moderate exercise (e.g., brisk walking) most days of the week.",
                "Reduce intake of saturated fats and cholesterol; eat more whole grains and vegetables.",
                "Limit salt intake to control blood pressure.",
                "Quit smoking and reduce alcohol consumption.",
                "Manage stress with mindfulness or relaxation techniques.",
                "Get regular medical check-ups to monitor heart health.",
            ])
        else:
            lifestyle_advice.append("Maintain your healthy lifestyle to continue minimizing heart disease risk.")
        return render_template("result.html", disease="Heart Disease", prediction=prediction,  features=data,lifestyle_advice=lifestyle_advice,read_more_info=disease_links["Heart Disease"])
    return render_template("heart.html")


@app.route("/cancer", methods=["GET", "POST"])
def cancer():
    if request.method == "POST":
        # Collect input data from the form
        data = [float(request.form[field]) for field in request.form]

        # Convert to numpy array
        input_array = np.array([data])

        # Make the prediction
        prediction = cancer_model.predict(input_array)[0]
        
        # Decode the prediction label (if you used LabelEncoder)
        prediction_label = label_encoder.inverse_transform([prediction])[0]
        # data = [float(request.form[field]) for field in request.form]
        # input_array = np.array([data])
        # prediction = cancer_model.predict(input_array)[0]
        
        
        
        #explanation = explainer_cancer(input_array)
        #shap_values=explanation.values[0].tolist() - remoeve this line if not using shap
        lifestyle_advice = []
        if prediction == 1:
            lifestyle_advice.extend([
                "Limit alcohol consumption and avoid smoking.",
                "Maintain a healthy weight through a nutritious diet and regular physical activity.",
                "Exercise for at least 150 minutes per week (moderate intensity).",
                "Consider regular breast self-exams and schedule clinical screenings.",
                "Avoid long-term hormone replacement therapy unless advised by a doctor.",
                "Increase intake of fiber, fruits, vegetables, and reduce red meat.",
            ])
        else:
            lifestyle_advice.append("Continue healthy habits and attend regular screenings to minimize cancer risk.")
        return render_template("result.html", disease="Breast Cancer Disease", prediction=prediction, features=data, lifestyle_advice=lifestyle_advice,read_more_info=disease_links["Breast Cancer Disease"])
    return render_template("cancer.html")



@app.route("/test", methods=["GET", "POST"])
def test_prediction():
    # Manually define test input (same format as your form data)
    test_data = [1,85,66,29,0,26.6,0.351,31,]  # Example values (update with real features)
    
    # Convert to numpy array (similar to how you're processing form data)
    input_array = np.array([test_data])
    
    # Get the model's prediction
    prediction = diabetes_model.predict(input_array)[0]
    
    # Print prediction to see what it outputs
    print("Test Prediction:", prediction)
    
    # Return the result to the page (for testing purposes)
    result = "Risk" if prediction == 1 else "No Risk"
    
    return f"Test Prediction: {prediction} -> {result}"
 #explanation = explainer_heart(input_array)
        #shap_values=explanation.values[0].tolist(),
        # Personalized lifestyle advice for heart disease

if __name__ == "__main__":
    app.run(debug=True)

