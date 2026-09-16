"""
Synthetic Data & RLCD Calibration Dataset Generator for OpenJudge.
Generates structured decision pairs with ground truth and ambiguity margins
to train and calibrate non-autoregressive decision heads.
"""

import json
import random
from typing import List, Dict, Any
from pydantic import BaseModel


class SyntheticSample(BaseModel):
    context: str
    question: str
    options: List[str]
    target_idx: int
    target_label: str
    is_ambiguous: bool
    ground_truth_probabilities: List[float]


class DecisionDatasetGenerator:
    """Generates synthetic domain datasets for classification, routing, and scoring."""

    ROUTING_TEMPLATES = [
        {
            "category": "security",
            "contexts": [
                "URGENT: Unauthorized SSH root access attempt detected from IP {ip}. Multiple failed attempts followed by sudo execution.",
                "Security Alert: CVE-2026-9921 exploited in cluster node {node}. Kernel memory dump initiated.",
                "DDoS Attack in progress: Ingress gateway receiving 4.2M rps on port 443 with SYN flood signature."
            ],
            "department": "security",
            "urgency": "critical",
            "is_fraud": False
        },
        {
            "category": "billing",
            "contexts": [
                "Invoice #9941 was charged twice to our corporate Amex card for $12,500. Please issue an immediate refund.",
                "We need to update our VAT tax ID for the upcoming Q3 enterprise billing cycle before Friday.",
                "Subscription cancellation request: Our budget was cut for Q4, please downgrade our plan to Free tier."
            ],
            "department": "billing",
            "urgency": "medium",
            "is_fraud": False
        },
        {
            "category": "fraud",
            "contexts": [
                "Wire transfer of $850,000 requested to offshore account in Seychelles without dual-signatory verification.",
                "Account takeover suspicious login from TOR exit node with immediate password change and crypto withdrawal.",
                "Chargeback spike: 45 transactions in 2 minutes using synthetic stolen Visa cards with CVV mismatch."
            ],
            "department": "security",
            "urgency": "critical",
            "is_fraud": True
        },
        {
            "category": "sales",
            "contexts": [
                "Hi team, we are an enterprise team of 500 engineers looking to deploy your platform on-premise next quarter.",
                "Can someone send us pricing tiers for 50,000 MAU and custom SLA agreements?",
                "Interested in scheduling a 30-min product demo with your solutions engineering team."
            ],
            "department": "sales",
            "urgency": "low",
            "is_fraud": False
        }
    ]

    @classmethod
    def generate_routing_samples(cls, n_samples: int = 100) -> List[Dict[str, Any]]:
        samples = []
        departments = ["billing", "security", "sales", "technical_support"]
        urgencies = ["low", "medium", "high", "critical"]

        for _ in range(n_samples):
            template = random.choice(cls.ROUTING_TEMPLATES)
            raw_ctx = random.choice(template["contexts"])
            ctx = raw_ctx.format(
                ip=f"192.168.{random.randint(1,254)}.{random.randint(1,254)}",
                node=f"node-prod-{random.randint(10,99)}"
            )

            # Department decision
            dept_target = template["department"]
            dept_idx = departments.index(dept_target)
            dept_probs = [0.05] * len(departments)
            dept_probs[dept_idx] = 0.85

            samples.append({
                "context": ctx,
                "question": "Which department should handle this request?",
                "options": departments,
                "target_idx": dept_idx,
                "target_label": dept_target,
                "probabilities": dept_probs
            })

            # Urgency decision
            urg_target = template["urgency"]
            urg_idx = urgencies.index(urg_target)
            urg_probs = [0.05] * len(urgencies)
            urg_probs[urg_idx] = 0.85

            samples.append({
                "context": ctx,
                "question": "What is the operational urgency level?",
                "options": urgencies,
                "target_idx": urg_idx,
                "target_label": urg_target,
                "probabilities": urg_probs
            })

        return samples


if __name__ == "__main__":
    data = DecisionDatasetGenerator.generate_routing_samples(10)
    print(f"Generated {len(data)} training pairs.")
    print("Sample:", json.dumps(data[0], indent=2))
