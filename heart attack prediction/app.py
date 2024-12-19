from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load('model.pkl')  # Ensure you've saved your model as 'model.pkl'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        data = request.form.to_dict()
        data = [float(data[key]) for key in data]
        data = [data]  # Make it a 2D array

        # Make prediction
        prediction = model.predict(data)
        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)