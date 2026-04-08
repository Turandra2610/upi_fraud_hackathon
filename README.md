---
title: UPI Fraud Detection Environment
emoji: 🛡️
colorFrom: indigo
colorTo: indigo
sdk: docker
app_port: 5000
tags:
  - openenv
  - pytorch
  - reinforcement-learning
---

# UPI Fraud Detection OpenEnv

A Reinforcement Learning environment built for the Meta AI Hackathon. This project simulates UPI transactions where an AI agent must learn to balance security (blocking fraud) and user experience (allowing legitimate users).

## 🚀 Quick Start

### 1. Build the Docker Image
From the project root directory:
```bash
docker build -t upi_fraud_env .