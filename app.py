from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

pickle_file_path = '/Users/siddartha/Desktop/github/code/Training_ML/gradient_boosting_model_with_oversampling.pkl'

# Load the model from the pickle file
with open(pickle_file_path, 'rb') as file:
    model = pickle.load(file)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    features = [float(x) for x in request.form.values()]
    
    # Make prediction
    prediction = model.predict([features])[0]
    
    # Convert prediction to human-readable format
    result = "IDA" if prediction == 1 else "NON-IDA"
    
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
