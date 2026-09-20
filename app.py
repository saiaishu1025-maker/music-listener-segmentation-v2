from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Cluster labels
cluster_names = {
    0: "Music Explorer",
    1: "Heavy Listener",
    2: "Casual Listener"
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    input_data = pd.DataFrame([{
        "listening_hours_per_week": float(data["listening_hours_per_week"]),
        "songs_per_day": float(data["songs_per_day"]),
        "skip_rate": float(data["skip_rate"]),
        "playlist_count": float(data["playlist_count"])
    }])

    # Scale input using the saved scaler
    input_scaled = scaler.transform(input_data)

    # Predict listener cluster
    cluster = int(model.predict(input_scaled)[0])

    return jsonify({
        "cluster": cluster,
        "segment": cluster_names.get(cluster, f"Cluster {cluster}")
    })


if __name__ == "__main__":
    app.run(debug=True)