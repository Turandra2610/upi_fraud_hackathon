from flask import Flask, request, jsonify
import torch
import os

# Import your specific class names from your files
from models import FraudPolicyNet
from upi_project_environment import UPIFraudEnv

app = Flask(__name__)

# Initialize your environment and model
env = UPIFraudEnv()
policy = FraudPolicyNet()

@app.route('/')
def home():
    return jsonify({"status": "online", "message": "UPI Fraud Defense API is running"})

@app.route('/reset', methods=['POST'])
def reset():
    obs, info = env.reset()
    # Ensure numpy arrays are converted to lists for JSON
    return jsonify({"observation": obs.tolist(), "info": info})

@app.route('/step', methods=['POST'])
def step():
    data = request.get_json()
    action = data.get('action', 0)
    obs, reward, done, truncated, info = env.step(action)
    return jsonify({
        "observation": obs.tolist(),
        "reward": reward,
        "done": bool(done or truncated),
        "info": info
    })

if __name__ == "__main__":
    # CRITICAL: Hugging Face Spaces MUST use port 7860
    app.run(host='0.0.0.0', port=7860)
