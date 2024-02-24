from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load your trained model
model = joblib.load("Data/best_model_reduced_10_fetures.pkl")

# Load test data
test_data = pd.read_csv("Data/test_data.csv")  # Adjust path as necessary
# Separate features and target variable
test_features = test_data.drop(['OP', 'Type'], axis=1)  # Drop 'OP' and 'Type' to isolate features
test_labels = test_data['OP']  # Actual labels

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_old', methods=['POST'])
def predict_old():
    data = request.get_json(force=True)
    prediction = model.predict([data['features']])
    return jsonify(prediction.tolist())

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = data['features']
    # Assuming the features are passed as a list in the correct order.
    # Convert features to the appropriate format expected by the model.
    prediction = model.predict(pd.DataFrame([features], columns=test_features.columns))
    return jsonify(prediction.tolist())

@app.route('/test_model', methods=['GET'])
def test_model():
    # Make predictions on the test features
    predictions = model.predict(test_features)
    
    # Calculate the prediction accuracy
    accuracy = accuracy_score(test_labels, predictions)
    
    return jsonify({'prediction_accuracy': accuracy})

@app.route('/upload', methods=['GET'])
def predict_first():
    first_row_features = test_features.iloc[0].values.reshape(1, -1)
    prediction = model.predict(first_row_features) 
    is_osteoporosis = prediction[0] == 1
    return jsonify({"Osteoporosis": is_osteoporosis})

if __name__ == '__main__':
    app.run(debug=True)
