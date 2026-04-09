# 🔐 UPI Fraud Pattern Investigator

An OpenEnv-compliant RL environment for detecting fraud patterns in Indian UPI transactions.

> India processes **14 billion UPI transactions per month**. Fraud is exploding. This environment simulates real-world fraud investigation scenarios across three difficulty levels.

---

## 🚀 Environment Overview

| Task | Difficulty | Description |
|------|-----------|-------------|
| `single-transaction-classify` | Easy | Classify one transaction: fraud / legitimate / needs-review |
| `account-compromise-detect` | Medium | Detect mule account patterns from 20 txns over 7 days |
| `fraud-ring-investigate` | Hard | Map a fraud ring across 15 accounts, find originator + cashout |

---

## 📐 Action Space

### Easy — Single Transaction Classification
```json
{
  "decision": "fraud | legitimate | needs-review",
  "confidence": 0.0,
  "fraud_type": "phishing | mule_account | sim_swap | null",
  "reasoning": "string"
}
```

### Medium — Account Compromise Detection
```json
{
  "is_compromised": true,
  "confidence": 0.85,
  "mule_pattern_detected": true,
  "suspicious_txn_ids": ["txn_abc123"],
  "reasoning": "string"
}
```

### Hard — Fraud Ring Investigation
```json
{
  "ring_detected": true,
  "confidence": 0.9,
  "originator_account": "ravi123@okaxis",
  "cashout_endpoint": "sneha456@ybl",
  "flagged_accounts": ["vpa1", "vpa2"],
  "money_flow_summary": "string",
  "reasoning": "string"
}
```

---

## 📊 Observation Space

Structured UPI transaction data including:
- Sender / receiver VPA (Virtual Payment Address)
- Transaction amount and timestamp
- Device fingerprint and location
- Merchant category and UPI app used
- Behavioural signals: velocity, amount deviation, new payee flag

---

## 🏆 Reward Function

| Scenario | Reward |
|----------|--------|
| Correct classification + high confidence | 0.8–1.0 |
| Correct label, wrong confidence | 0.5–0.7 |
| Adjacent error (e.g. fraud → needs-review) | 0.2–0.3 |
| Wrong label with high confidence | 0.0–0.1 |

**False positive penalty** (blocking legit users) and **false negative penalty** (missing fraud) are both built into the reward function — creating the core tension of fraud detection.

---

## 🛠️ Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run inference baseline
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start new episode (optionally specify task + seed) |
| `POST` | `/step` | Submit agent action, receive reward |
| `GET` | `/state` | Get current episode state |
| `GET` | `/tasks` | List all available tasks |
| `GET` | `/health` | Health check |

---

## 🐳 Docker

```bash
docker build -t upi-fraud-investigator .
docker run -p 7860:7860 upi-fraud-investigator
```

---

## 📋 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `Qwen/Qwen2.5-72B-Instruct` |
| `HF_TOKEN` | HuggingFace API key | — |
| `ENV_BASE_URL` | Environment server URL | `http://localhost:7860` |

---

## 🇮🇳 Why UPI Fraud?

- **Scale**: 14B transactions/month, growing 40% YoY
- **Unique fraud patterns**: SIM swaps, mule networks, round-tripping — all India-specific
- **Real stakes**: False positives block real users; false negatives lose real money
- **Directly relevant**: Meta Pay / WhatsApp Pay operates in this exact ecosystem
