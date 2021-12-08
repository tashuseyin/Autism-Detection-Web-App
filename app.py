# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

filename = 'random_forest_model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('main.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = request.form.get('sex')
        a1 = request.form.get('A1')
        a2 = request.form.get('A2')
        a3 = request.form.get('A3')
        a4 = request.form.get('A4')
        a5 = request.form.get('A5')
        a6 = request.form.get('A6')
        a7 = request.form.get('A7')
        a8 = request.form.get('A8')
        a9 = request.form.get('A9')
        a10 = request.form.get('A10')
        Ethnicity = request.form.get('Ethnicity')
        Jaundice = request.form.get('Jaundice')
        Family_mem_with_ASD = request.form.get('Family_mem_with_ASD')
        data = np.array([[age, sex, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, Ethnicity, Jaundice, Family_mem_with_ASD]])
        my_prediction = model.predict(data)

        return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
