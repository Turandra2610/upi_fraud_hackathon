# UPI Fraud Investigation RL Environment

### Overview
This is a custom Reinforcement Learning environment built for the Meta PyTorch OpenEnv Hackathon. It simulates real-world UPI transaction patterns to train agents in identifying and preventing fraudulent activities.

### Environment Logic
- **Observations:** Transaction amount, sender frequency (velocity), and device risk score.
- **Actions:** - `Allow`: Process the transaction.
    - `Flag`: Send for secondary verification.
    - `Block`: Immediate termination of high-risk transactions.

### Reward System
- **+10**: Successfully blocking a fraudulent transaction.
- **+1**: Correctly allowing a safe transaction.
- **-2**: False Positive (Blocking a legitimate user).
- **-10**: False Negative (Allowing a high-value fraud).
