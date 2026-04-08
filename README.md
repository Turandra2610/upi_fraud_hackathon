---
title: UPI Fraud Detection Environment
emoji: 🛡️
colorFrom: indigo
colorTo: indigo
sdk: docker
app_port: 7860
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
`docker build -t upi_fraud_env .`

### 2. Run the Container
`docker run -p 7860:7860 upi_fraud_env`
