from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Se carga el modelo de machine learning previamente entrenado

modelo = joblib.load('best_model_dt_2.joblib')

@app.route('/') # Index
def formulario():
    return render_template('formulario.html')

@app.route('/inferencia', methods=['POST'])
def realizar_inferencia():
    try:
        # Datos del formulario
        form_data = request.form.to_dict()

        # Entrada del modelo (datos para hacer inferencia)
        X_test = np.array(list(form_data.values())).reshape(1, -1)

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
        return jsonify({'error': str(e)})

if __name__ == '__main__':
   #app.run(port=8010)
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port)
