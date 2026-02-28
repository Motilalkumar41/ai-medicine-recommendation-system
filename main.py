# from flask import Flask, request, render_template, jsonify  # Import jsonify
# import numpy as np
# import pandas as pd
# import pickle


# # flask app
# app = Flask(__name__)



# # load databasedataset===================================
# sym_des = pd.read_csv("datasets/symtoms_df.csv")
# precautions = pd.read_csv("datasets/precautions_df.csv")
# workout = pd.read_csv("datasets/workout_df.csv")
# description = pd.read_csv("datasets/description.csv")
# medications = pd.read_csv('datasets/medications.csv')
# diets = pd.read_csv("datasets/diets.csv")


# # load model===========================================
# svc = pickle.load(open('models/svc.pkl','rb'))


# #============================================================
# # custome and helping functions
# #==========================helper funtions================
# def helper(dis):
#     desc = description[description['Disease'] == dis]['Description']
#     desc = " ".join([w for w in desc])

#     pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
#     pre = [col for col in pre.values]

#     med = medications[medications['Disease'] == dis]['Medication']
#     med = [med for med in med.values]

#     die = diets[diets['Disease'] == dis]['Diet']
#     die = [die for die in die.values]

#     wrkout = workout[workout['disease'] == dis] ['workout']


#     return desc,pre,med,die,wrkout

# symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
# diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# # Model Prediction function
# def get_predicted_value(patient_symptoms):
#     input_vector = np.zeros(len(symptoms_dict))
#     for item in patient_symptoms:
#         input_vector[symptoms_dict[item]] = 1
#     return diseases_list[svc.predict([input_vector])[0]]




# # creating routes========================================


# @app.route("/")
# def index():
#     return render_template("index.html")

# # Define a route for the home page
# @app.route('/predict', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         symptoms = request.form.get('symptoms')
#         # mysysms = request.form.get('mysysms')
#         # print(mysysms)
#         print(symptoms)
#         if symptoms =="Symptoms":
#             message = "Please either write symptoms or you have written misspelled symptoms"
#             return render_template('index.html', message=message)
#         else:

#             # Split the user's input into a list of symptoms (assuming they are comma-separated)
#             user_symptoms = [s.strip() for s in symptoms.split(',')]
#             # Remove any extra characters, if any
#             user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
#             predicted_disease = get_predicted_value(user_symptoms)
#             dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

#             my_precautions = []
#             for i in precautions[0]:
#                 my_precautions.append(i)

#             return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
#                                    my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
#                                    workout=workout)

#     return render_template('index.html')



# # about view funtion and path
# @app.route('/about')
# def about():
#     return render_template("about.html")
# # contact view funtion and path
# @app.route('/contact')
# def contact():
#     return render_template("contact.html")

# # developer view funtion and path
# @app.route('/developer')
# def developer():
#     return render_template("developer.html")

# # about view funtion and path
# @app.route('/blog')
# def blog():
#     return render_template("blog.html")


# if __name__ == '__main__':

#     app.run(debug=True)

# ==========================================================
# Medicine Recommendation System (Flask App)
# Path-robust + includes full symptoms_dict & diseases_list
# ==========================================================

from flask import Flask, request, render_template
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# ------------------------
# BASE PATH
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def p(*paths):
    return os.path.join(BASE_DIR, *paths)

def read_csv_any(*candidates):
    for c in candidates:
        full = p(c)
        if os.path.exists(full):
            return pd.read_csv(full)
    raise FileNotFoundError(f"CSV not found: {candidates}")

def load_any(*candidates):
    for c in candidates:
        full = p(c)
        if os.path.exists(full):
            return full
    raise FileNotFoundError(f"Model not found: {candidates}")

# ------------------------
# FLASK APP
# ------------------------
app = Flask(__name__, template_folder='.', static_folder='static')

# ------------------------
# LOAD DATA (SAFE)
# ------------------------
training_df = read_csv_any(
    "Training_medical_FULL_clean_FINAL.csv"

)

description = read_csv_any(
    "description_clean_final.csv"
    
)

precautions = read_csv_any(
    "precaution_clean_final.csv"
    
)

medications = read_csv_any(
    "medication_clean_final.csv"
    
)

diets = read_csv_any(
    "diet_clean_final.csv"
)

try:
    workout = read_csv_any(
        "workout_clean_final.csv"
        
    )
except:
    workout = pd.DataFrame(columns=["disease", "workout"])

# ------------------------
# LABEL ENCODER
# ------------------------
le = LabelEncoder()
le.fit(training_df["prognosis"])
diseases_list = {i: d for i, d in enumerate(le.classes_)}

# ------------------------
# LOAD MODEL
# ------------------------
svc_path = load_any("models/svc.pkl", "svc.pkl")
with open(svc_path, "rb") as f:
    svc = pickle.load(f)

# ------------------------
# COMBINATION → SYMPTOMS
# ------------------------
COMBO_SYMPTOMS = {
    "malaria": ["high_fever", "chills", "sweating", "headache"],
    "dengue": ["high_fever", "joint_pain", "pain_behind_eyes", "nausea"],
    "viral_fever": ["high_fever", "fatigue", "body_pain"],
    "typhoid": ["high_fever", "abdominal_pain", "constipation"],
    "common_cold": ["sneezing", "runny_nose", "sore_throat"],
    "migraine": ["headache", "nausea", "vomiting", "sensitivity_to_light"],
    "food_poisoning": ["vomiting", "diarrhea", "abdominal_pain"],
    "gastritis": ["stomach_pain", "acidity", "nausea"],
    "pneumonia": ["cough", "chest_pain", "breathing_problem", "high_fever"],
    "tuberculosis": ["cough", "chest_pain", "weight_loss", "night_sweats"],
    "arthritis": ["joint_pain", "stiffness", "swelling"],
    "anxiety": ["anxiety", "restlessness", "sweating"],
    "Hyperthyroidism": ["sleeplessness", "fatigue", "irritability"]
}

# ------------------------
# HELPER (SAFE)
# ------------------------
def helper(dis):
    desc = " ".join(
        description[description["Disease"] == dis]["Description"].astype(str)
    )

    pre_df = precautions[precautions["Disease"] == dis]
    if not pre_df.empty:
        pre = pre_df[['Precaution_1','Precaution_2','Precaution_3','Precaution_4']].values.tolist()
    else:
        pre = [["No precautions data available"]]

    med = medications[medications["Disease"] == dis]["Medication"].tolist()
    die = diets[diets["Disease"] == dis]["Diet"].tolist()

    wrk = []
    if not workout.empty:
        wrk = workout[workout["disease"] == dis]["workout"].tolist()

    return desc, pre, med, die, wrk

# ------------------------
# PREDICTION LOGIC
# ------------------------
def predict_disease(symptoms):

    if hasattr(svc, "feature_names_in_"):
        features = list(svc.feature_names_in_)
    else:
        features = list(training_df.columns[:-1])

    x = np.zeros(len(features), dtype=int)

    for s in symptoms:
        if s == "fever":
            s = "high_fever"
        if s in features:
            x[features.index(s)] = 1

    predicted_index = svc.predict([x])[0]
    predicted = diseases_list[predicted_index]

    # TB safety rule
    if predicted == "Tuberculosis":
        if "wheezing" in symptoms and "high_fever" not in symptoms:
            return "⚠️ Please add fever-related symptoms"

    return predicted


    predicted = diseases_list[np.argmax(probs)]

    # TB safety rule
    # if predicted == "Tuberculosis":
    #     if "wheezing" in symptoms and "high_fever" not in symptoms:
    #         return "⚠️ Please add fever-related symptoms"

    # return predicted

# ------------------------
# ROUTES
# ------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    combo = request.form.get("combo")

    if not combo or combo not in COMBO_SYMPTOMS:
        return render_template(
            "index.html",
            message="⚠️ Please select a symptom combination"
        )

    symptoms = COMBO_SYMPTOMS[combo]
    predicted_disease = predict_disease(symptoms)

    if predicted_disease is None or str(predicted_disease).startswith("⚠️"):
        return render_template(
            "index.html",
            message=predicted_disease or
            "⚠️ Not enough information. Please try another combination."
        )

    dis_des, pre, med, die, wrk = helper(predicted_disease)

    return render_template(
        "index.html",
        predicted_disease=predicted_disease,
        dis_des=dis_des,
        my_precautions=pre[0],
        medications=med,
        my_diet=die,
        workout=wrk
    )

# ------------------------
# RUN
# ------------------------
if __name__ == "__main__":
    app.run(debug=True)

    

