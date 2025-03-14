from flask import Flask, jsonify
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

predictor = pickle.load(open("Models/model_xgb.pkl", "rb"))
scaler = pickle.load(open("Models/scaler.pkl", "rb"))
cv = pickle.load(open("Models/countVectorizer.pkl", "rb"))

STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()

app = Flask(__name__)

def preprocess_text(text):
    review = re.sub("[^a-zA-Z]", " ", text)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    review = " ".join(review)
    return review

def predict_sentiment(text):
    """Predict sentiment (Positive/Negative) of the input text."""
    processed_text = preprocess_text(text)
    X_prediction = cv.transform([processed_text]).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_prediction = predictor.predict_proba(X_prediction_scl)

    if y_prediction[0][1] > 0.6:  
        return "Positive"
    elif y_prediction[0][0] > 0.5:  
        return "Negative"
    else:
        return "Neutral"


@app.route("/predict/<text>", methods=["GET"])
def predict(text):
    """API endpoint to return sentiment prediction."""
    sentiment = predict_sentiment(text)
    return jsonify({"prediction": sentiment})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
