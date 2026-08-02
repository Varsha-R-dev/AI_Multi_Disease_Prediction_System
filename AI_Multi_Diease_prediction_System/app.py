from flask import Flask, render_template, request,redirect,session
import mysql.connector
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key="mysecretkey"

# ----------------------------
# MySQL Connection
# ----------------------------
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="multi_disease_prediction1"
)

cursor = db.cursor()

# ----------------------------
# Load Diabetes ML Model
# ----------------------------
diabetes_model = pickle.load(open("models/diabetes_model.pkl", "rb"))
heart_model=pickle.load(open("models/heart_model.pkl","rb"))
liver_model=pickle.load(open("models/liver_model.pkl","rb"))
kidney_model=pickle.load(open("models/kidney_model.pkl","rb"))
parkinsons_model=pickle.load(open("models/parkinsons_model.pkl","rb"))


# ----------------------------
# Home Page
# ----------------------------
@app.route('/')
def home():
    return render_template("index.html")


# ----------------------------
# Register
# ----------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():

    if request.method == 'POST':

        fullname = request.form['fullname']
        email = request.form['email']
        password = request.form['password']

        sql = "INSERT INTO users(full_name,email,password) VALUES(%s,%s,%s)"
        values = (fullname, email, password)

        cursor.execute(sql, values)
        db.commit()

        return '''
        <h2>Registration Successful!</h2>
        <a href="/login">Click Here to Login</a>
        '''

    return render_template("register.html")


# ----------------------------
# Login
# ----------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':

        email = request.form['email']
        password = request.form['password']

        sql = "SELECT * FROM users WHERE email=%s AND password=%s"
        values = (email, password)

        cursor.execute(sql, values)

        user = cursor.fetchone()

        if user:
           session["email"]=email
           return redirect("/dashboard")
        else:
            return '''
            <h2>Invalid Email or Password!</h2>
            <a href="/login">Try Again</a>
            '''

    return render_template("login.html")


# ----------------------------
# Dashboard
# ----------------------------
@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")


@app.route("/history")
def history():

    cursor.execute(
        "SELECT disease,prediction,prediction_date FROM prediction_history WHERE username=%s",
        (session["email"],)
    )

    records = cursor.fetchall()

    return render_template("history.html", records=records)


# ----------------------------
# Diabetes Prediction
# ----------------------------
@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():

    prediction = ""

    if request.method == 'POST':

        input_data = np.array([[
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]])

        result = diabetes_model.predict(input_data)

        if result[0] == 1:
            prediction = "⚠️ You are likely Diabetic."
        else:
            prediction = "✅ You are Not Diabetic."

        cursor.execute(
            "INSERT INTO prediction_history(user_email,disease_name,prediction_result) VALUES(%s,%s,%s)",
            (session["email"], "Diabetes", prediction)
        )  
        db.commit()

    return render_template("diabetes.html", prediction=prediction)


# ----------------------------
# Heart Prediction
# ----------------------------
@app.route('/heart', methods=['GET', 'POST'])
def heart():

    prediction = ""

    if request.method == 'POST':  
        input_data = np.array([[
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal']),
        ]])

        result = heart_model.predict(input_data)

        if result[0] == 1:
            prediction = "⚠️ You are likely Affected Heart Disease."
        else:
            prediction = "✅ You are Not Affected."
        
        cursor.execute(
            "INSERT INTO prediction_history(user_email,disease_name,prediction_result) VALUES(%s,%s,%s)",
            (session["email"], "heart", prediction)
        )  
        db.commit()

    return render_template("heart.html", prediction=prediction)



# -----------------------------
# Liver Disease Prediction
# -----------------------------

@app.route('/liver', methods=['GET', 'POST'])
def liver():

    prediction = ""

    if request.method == 'POST':

        input_data = np.array([[
            float(request.form['Age']),
            float(request.form['Gender']),
            float(request.form['Total_Bilirubin']),
            float(request.form['Direct_Bilirubin']),
            float(request.form['Alkaline_Phosphotase']),
            float(request.form['Alamine_Aminotransferase']),
            float(request.form['Aspartate_Aminotransferase']),
            float(request.form['Total_Protiens']),
            float(request.form['Albumin']),
            float(request.form['Albumin_and_Globulin_Ratio'])
        ]])

        result = liver_model.predict(input_data)

        if result[0] == 1:
            prediction = "⚠ Liver Disease Detected."
        else:
            prediction = "✅ No Liver Disease Detected."

        cursor.execute(
            "INSERT INTO prediction_history(user_email,disease_name,prediction_result) VALUES(%s,%s,%s)",
            (session["email"], "liver", prediction)
        )  
        db.commit()

    return render_template("liver.html", prediction=prediction)

# -----------------------------
# Kidney Disease Prediction
# -----------------------------

@app.route("/kidney", methods=["GET", "POST"])
def kidney():

    prediction = ""

    if request.method == "POST":

        input_data = np.array([[
            float(request.form["id"]),
            float(request.form["age"]),
            float(request.form["bp"]),
            float(request.form["sg"]),
            float(request.form["al"]),
            float(request.form["su"]),
            float(request.form["rbc"]),
            float(request.form["pc"]),
            float(request.form["pcc"]),
            float(request.form["ba"]),
            float(request.form["bgr"]),
            float(request.form["bu"]),
            float(request.form["sc"]),
            float(request.form["sod"]),
            float(request.form["pot"]),
            float(request.form["hemo"]),
            float(request.form["pcv"]),
            float(request.form["wc"]),
            float(request.form["rc"]),
            float(request.form["htn"]),
            float(request.form["dm"]),
            float(request.form["cad"]),
            float(request.form["appet"]),
            float(request.form["pe"]),
            float(request.form["ane"])
        ]])

        result = kidney_model.predict(input_data)

        if result[0] == 1:
            prediction = "⚠ Kidney Disease Detected"
        else:
            prediction = "✅ No Kidney Disease"

        cursor.execute(
            "INSERT INTO prediction_history(user_email,disease_name,prediction_result) VALUES(%s,%s,%s)",
            (session["email"], "kidney", prediction)
        )  
        db.commit()

    return render_template("kidney.html", prediction=prediction)

# -----------------------------
# Parkinsons Disease Prediction
# -----------------------------

@app.route("/parkinsons", methods=["GET", "POST"])
def parkinsons():

    prediction = ""

    if request.method == "POST":

        input_data = np.array([[
            float(request.form["fo"]),
            float(request.form["fhi"]),
            float(request.form["flo"]),
            float(request.form["jitter"]),
            float(request.form["jitter_abs"]),
            float(request.form["rap"]),
            float(request.form["ppq"]),
            float(request.form["ddp"]),
            float(request.form["shimmer"]),
            float(request.form["shimmer_db"]),
            float(request.form["apq3"]),
            float(request.form["apq5"]),
            float(request.form["apq"]),
            float(request.form["dda"]),
            float(request.form["nhr"]),
            float(request.form["hnr"]),
            float(request.form["rpde"]),
            float(request.form["dfa"]),
            float(request.form["spread1"]),
            float(request.form["spread2"]),
            float(request.form["d2"]),
            float(request.form["ppe"])
        ]])

        result = parkinsons_model.predict(input_data)

        if result[0] == 1:
            prediction = "⚠ Parkinson's Disease Detected"
        else:
            prediction = "✅ No Parkinson's Disease"

        cursor.execute(
            "INSERT INTO prediction_history(user_email,disease_name,prediction_result) VALUES(%s,%s,%s)",
            (session["email"], "parkisons", prediction)
        )  
        db.commit()

    return render_template("parkinsons.html", prediction=prediction)



# ----------------------------
# Logout
# ----------------------------
@app.route('/logout')
def logout():
    return render_template("index.html")


# ----------------------------
# Run Application
# ----------------------------
if __name__ == '__main__':
    app.run(debug=True)