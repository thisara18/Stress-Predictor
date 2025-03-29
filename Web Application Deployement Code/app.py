from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('best_dt_model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')

@app.route('/')
def home():
    return render_template('index.html', selected_features=selected_features)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the form
    user_input = []
    
    for feature in selected_features:
        feature_value = float(request.form[feature])
        user_input.append(feature_value)
    
    # Convert user input to a numpy array and scale it
    user_input = np.array(user_input).reshape(1, -1)
    user_input_scaled = scaler.transform(user_input)

    # Make the prediction using the trained model
    prediction = model.predict(user_input_scaled)

    # Determine the message and color based on prediction
    if prediction[0] == 0:
        prediction_text = "Your stress level is low."
        stress_instruction = "Try to keep up with your healthy routine, stay positive!"
        result_color = "green"
    elif prediction[0] == 1:
        prediction_text = "Your stress level is medium."
        stress_instruction = "Consider relaxing activities, such as meditation or talking to someone."
        result_color = "orange"
    else:
        prediction_text = "Your stress level is high."
        stress_instruction = "It’s important to seek support. Consider talking to a counselor."
        result_color = "red"

    return render_template('index.html', prediction_text=prediction_text, 
                           stress_instruction=stress_instruction, 
                           result_color=result_color, 
                           selected_features=selected_features)

if __name__ == "__main__":
    app.run(debug=True)
