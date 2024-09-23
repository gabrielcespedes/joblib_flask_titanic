from flask import Flask, render_template, request
import joblib
import numpy as np

#Inicializar Flask App
app = Flask(__name__)

#Cargar el modelo previamente guardado
model = joblib.load('modelo_titanic.joblib')

#Ruta para la página principal
@app.route('/', methods = ['GET'])
def home():
    return render_template('index.html')

#Ruta para predecir la supervivencia
@app.route('/predict', methods=['POST'])
def predict():
    pclass = int(request.form['pclass'])
    age = float(request.form['age'])
    fare = float(request.form['fare'])
    sex = int(request.form['sex']) # 0 para female, 1 para male
    embarked = int(request.form['embarked'])  # 0 para Cherbourg, 1 para Queenstown, 2 para Southampton
    alone = int(request.form['alone']) # 0 si no viaja solo, 1 si viaja solo

    # Transformar variables categóricas en las dummies esperadas por el modelo
    sex_male = 1 if sex == 1 else 0  # 1 si es hombre, 0 si es mujer
    embarked_Q = 1 if embarked == 1 else 0  # Queenstown
    embarked_S = 1 if embarked == 2 else 0  # Southampton
    alone_true = 1 if alone == 1 else 0  # True si viaja solo, 0 si no

    # Crear el array de entrada para el modelo con las 7 características
    features = np.array([[pclass,age, fare, sex_male, embarked_Q, embarked_S, alone_true]])

    #Realizar predicción
    prediction = model.predict(features)
    print("PREDICCIÓN:", prediction)

    #Interpretar resultado predicción
    result = 'Sobrevivió' if prediction[0] == 1 else 'No sobrevivió'

    #Renderizar el resultado en la página
    return render_template('index.html', result = result)

if __name__ == "__main__":
    app.run(port=5000)
