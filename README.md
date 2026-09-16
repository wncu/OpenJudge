# OpenJev: High-Performance System-One Decision Engine

> **Non-autoregressive, parallel, calibrated decision intelligence for software automation.**
> Built as an open-source alternative to proprietary "System-One" decision models like Jev / TypeSafe AI.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Type Safe: Pydantic](https://img.shields.io/badge/Type%20Safe-Pydantic%20v2-green.svg)](https://docs.pydantic.dev/)

---

## ⚡ The Problem with LLMs in Automated Software

When developers use LLMs (GPT-4, Claude, Llama) for decision logic or JSON extraction:
1. **Autoregressive bottleneck:** Generating `{ "status": "approved", "confidence": 0.94 }` requires **50 to 150 sequential forward passes**.
2. **High Latency & High Cost:** 1.5s to 6s per call, charging heavy output token premiums.
3. **Miscalibration & Hallucination:** RLHF induces overconfidence, missing keys, and invalid types.

## 🚀 The OpenJev Solution

OpenJev replaces string generation with **parallel non-autoregressive logits evaluation**:

- ⚡ **10x–100x Faster:** Evaluates all decision dimensions in a **single forward pass** (15ms–80ms).
- 🛡️ **0% Type Errors:** Schema outputs are mathematically bounded to your Pydantic model definitions.
- 🎯 **Calibrated Uncertainty:** Returns actual probability distributions and Shannon Entropy for every field.
- 💸 **Zero Output Token Billing:** Evaluates logits directly from candidate tokens without autoregressive decoding.

---

## 🛠️ Quickstart

### 1. Installation

```bash
git clone https://github.com/<your-username>/OpenJev.git
cd OpenJev
pip install -e .
```

Optional acceleration dependencies:
```bash
pip install "openjev[torch,onnx,server]"
```

### 2. Define Decision Schema & Run

```python
from enum import Enum
from pydantic import BaseModel
from openjev import OpenJevEngine, DecisionField

class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SupportTriageDecision(BaseModel):
    is_spam: bool = DecisionField("Is this ticket promotional spam or bot noise?")
    is_escalated: bool = DecisionField("Does the customer express anger or cancel intent?")
    urgency: UrgencyLevel = DecisionField("Operational urgency level")

# Initialize Engine
engine = OpenJevEngine()

context = "URGENT: Production database locked! Charging error duplicated $4500 on our account. Fix now or we cancel."

# Single-pass non-autoregressive decision
result = engine.decide(context=context, schema=SupportTriageDecision)

print(result.values)
# {'is_spam': False, 'is_escalated': True, 'urgency': 'critical'}

print(f"Latency: {result.latency_ms} ms")
# Latency: 24.18 ms

# Cast directly to verified Pydantic model
model_instance = result.get_typed_instance(SupportTriageDecision)
```

---

## 🔬 Mathematical Calibration (RLCD Alternative)

OpenJev implements post-hoc calibration methods (Temperature Scaling & Platt Scaling) that minimize **Expected Calibration Error (ECE)**:

$$\hat{p}_i = \frac{e^{z_i / T}}{\sum_{j=1}^K e^{z_j / T}}$$

With **Shannon Entropy** calculation on every decision:

$$H(P) = - \sum_{i=1}^K p_i \ln(p_i)$$

If $H(P)$ exceeds your safety threshold, your code can seamlessly route ambiguous edge-cases to human review or a fallback model.

---

## 🌐 Running as a Microservice

OpenJev includes a high-throughput FastAPI service:

```bash
python -m openjev.server
# or
uvicorn openjev.server:app --workers 4 --port 8000
```

### Endpoint: `POST /decide`

```json
{
  "context": "Customer email text...",
  "fields": [
    {
      "name": "is_fraud",
      "description": "Is this transaction fraudulent?",
      "options": ["True", "False"],
      "is_boolean": true
    }
  ],
  "temperature": 1.0
}
```

---

## 📊 Benchmark & Architecture

Read the full technical deep dive in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## 📄 License
MIT License. Free for personal and commercial automation infrastructure.
