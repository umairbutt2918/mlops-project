from flask import Flask, render_template, request, jsonify, Response
import pickle, os, logging
import models_utils
from prometheus_client import Counter, generate_latest

app = Flask(__name__)

# --- MLOps: Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- MLOps: Prometheus Metrics ---
PREDICTION_COUNTER = Counter('sentiment_predictions_total', 'Total sentiment predictions', ['model', 'sentiment'])

BASE_DIR = os.path.dirname(__file__)
LOG_MODEL = os.path.join(BASE_DIR, "logistic_regression_model.pkl")
GB_MODEL = os.path.join(BASE_DIR, "gradient_boosting_model.pkl")
VEC = os.path.join(BASE_DIR, "vectorizer.pkl")

models = {"log": None, "gb": None, "vec": None}

def load_models():
    try:
        if models["vec"] is None:
            with open(VEC, "rb") as f: models["vec"] = pickle.load(f)
        if models["log"] is None and os.path.exists(LOG_MODEL):
            with open(LOG_MODEL, "rb") as f: models["log"] = pickle.load(f)
        if models["gb"] is None and os.path.exists(GB_MODEL):
            with open(GB_MODEL, "rb") as f: models["gb"] = pickle.load(f)
        return True, None
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return False, str(e)

@app.route("/", methods=["GET","POST"])
def index():
    if request.method=="POST":
        review = request.form.get("review", "")
        model_choice = request.form.get("model", "Logistic Regression")
        
        ok, err = load_models()
        if not ok: return render_template("index.html", error="Error loading models.")
        
        try:
            X = models["vec"].transform([review])
            model = models["gb"] if model_choice=="Gradient Boosting" else models["log"]
            
            if model is None:
                return render_template("index.html", error="Selected model not found on server.")
                
            pred = int(model.predict(X)[0])
            sentiment = "Positive 😄" if pred==1 else "Negative 😞"
            
            # --- MLOps: Update Metrics ---
            PREDICTION_COUNTER.labels(model=model_choice, sentiment="positive" if pred==1 else "negative").inc()
            logger.info(f"Prediction: {sentiment} | Model: {model_choice} | Input Length: {len(review)}")

            return render_template("index.html", sentiment=sentiment, review=review)
        except Exception as e:
            logger.error(f"Prediction Error: {e}")
            return render_template("index.html", error=f"Error: {e}")
            
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    review = data.get("review","")
    model_choice = data.get("model","Logistic Regression")
    
    ok, err = load_models()
    if not ok: return jsonify({"error": err}), 500
    
    try:
        X = models["vec"].transform([review])
        model = models["gb"] if model_choice=="Gradient Boosting" else models["log"]
        pred = int(model.predict(X)[0])
        sentiment = "positive" if pred==1 else "negative"
        
        PREDICTION_COUNTER.labels(model=model_choice, sentiment=sentiment).inc()
        
        return jsonify({"sentiment": sentiment})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- MLOps: Metrics Endpoint for Prometheus ---
@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype='text/plain')

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)
