from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Se carga el modelo de machine learning previamente entrenado

modelo = joblib.load('best_model_rf.joblib')

# Se carga el encoder previamente entrenado
encoder = joblib.load('encoder_train.joblib')

@app.route('/') # Index
def formulario():
    try:
        return render_template('Formulario.html')
    except Exception as e:

        return jsonify({'error': str(e)})

@app.route('/inferencia', methods=['POST'])
def realizar_inferencia():
    try:
  
        # Datos del formulario
        form_data = request.form.to_dict()

	# Preparación datos para la inferencia
        values_inference = pd.DataFrame.from_dict(form_data, orient='index').transpose()

	    # Se aplica el encoder a las variables categóricas
        categorical_variables = ['Occupation', 'Type_of_Loan','Credit_Mix','Payment_Behaviour','Payment_of_Min_Amount'] 

        values_inference_categorical = values_inference[categorical_variables]
        
        # Se aplica el encoder a todas las columnas categóricas a la vez
        values_inference_categorical_encoded = encoder.transform(values_inference_categorical)  

        # Se reemplaza las columnas categóricas originales con las codificadas
        values_inference[categorical_variables] = values_inference_categorical_encoded

        # Entrada del modelo (datos para hacer inferencia)
        #X_test = np.array(list(form_data.values())).reshape(1, -1)

	    # Entrada del modelo (datos para hacer inferencia)
        X_test = values_inference.values.reshape(1, -1)
        X_test = pd.DataFrame(X_test,columns=values_inference.columns)

        # Se realiza la inferencia con el modelo
        inference = modelo.predict(X_test)

        credit_scoring = None

        if(inference)==0:
             credit_scoring="Poor"
        elif(inference)==1:
             credit_scoring="Standard"
        else:
             credit_scoring="Good"

        # Resultado de la inferencia en negrita
        credit_scoring = f'<span style="font-size: larger;"><b>Credit Scoring:</b> {credit_scoring}</span>'
        return credit_scoring
    
    except Exception as e:
        print("error")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    
    app.run()
    print("corriendo")
