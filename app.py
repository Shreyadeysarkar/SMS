from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model
with open('spam_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    # Transform the message using the loaded vectorizer
    message_transformed = vectorizer.transform([message])
    prediction = model.predict(message_transformed)
    return render_template('index.html', prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
