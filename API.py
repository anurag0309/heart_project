from flask import Flask,render_template,request
import pickle

app = Flask(__name__)

model = pickle.load(open('/knn_model_hpp.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')
    # return "SUCCESS"

@app.route('/predict' ,methods = ["POST","GET"])
def student():
    age= int(request.form.get('age'))
    sex= int(request.form.get('sex'))
    cp= int(request.form.get('cp'))
    trestbps= int(request.form.get('trestbps'))
    chol= int(request.form.get('chol'))
    fbs= int(request.form.get('fbs'))
    restecg= int(request.form.get('restecg'))
    thalach= int(request.form.get('thalach'))
    exang= int(request.form.get('exang'))
    oldpeak= float(request.form.get('oldpeak'))
    slope= int(request.form.get('slope'))
    ca= int(request.form.get('ca'))
    thal= int(request.form.get('thal'))

    # print(f"{var_cgpa},{var_iq},{var_ps}")

    result = model.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    print(result[0])

    return render_template('index.html',prediction = result[0])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0' , port=8080)