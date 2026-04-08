from flask import Flask, request, jsonify
import sys
import os

# Add root to path so Flask can find models.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import FraudPolicy
from upi_project_environment import UPIFraudEnv

app = Flask(__name__)
env = UPIFraudEnv()
policy = FraudPolicy()

@app.route('/reset', methods=['POST'])
def reset():
    obs, info = env.reset()
    return jsonify({"observation": obs.tolist(), "info": info})

@app.route('/step', methods=['POST'])
def step():
    action = request.json.get('action', 0)
    obs, reward, done, truncated, info = env.step(action)
    return jsonify({
        "observation": obs.tolist(),
        "reward": reward,
        "done": done or truncated,
        "info": info
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    