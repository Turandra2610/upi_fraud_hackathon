from flask import Flask, request, jsonify
import torch
import os
from models import FraudPolicyNet
from upi_project_environment import UPIFraudEnv

app = Flask(__name__)

env = UPIFraudEnv()
policy = FraudPolicyNet()

@app.route('/')
def home():
    return jsonify({"status": "online", "message": "UPI Fraud Defense API is running"})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/reset', methods=['POST'])
def reset():
    obs, info = env.reset()
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
    app.run(host='0.0.0.0', port=7860)
