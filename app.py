
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load vectorizer and model
vectorizer = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))
nb_model = pickle.load(open("model/naive_bayes_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')  # Serve frontend

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or "email" not in data:
        return jsonify({"error": "Please provide an 'email' key in the JSON request"}), 400

    email_text = data["email"]
    email_vector = vectorizer.transform([email_text])

    # Prediction
    nb_prediction = nb_model.predict(email_vector)[0]
    nb_result = "Spam" if nb_prediction == 1 else "Not Spam"

    return jsonify({
        "email": email_text,
        "Naïve Bayes Prediction": nb_result
    })

if __name__ == '__main__':
    app.run(debug=True)

