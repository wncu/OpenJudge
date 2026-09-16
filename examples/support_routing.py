from enum import Enum
from pydantic import BaseModel
from openjev import OpenJevEngine, DecisionField

class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Department(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical_support"
    SALES = "sales"
    SECURITY = "security"

class TicketTriageDecision(BaseModel):
    is_spam: bool = DecisionField(description="Is this customer support ticket promotional spam, advertising, or automated bot noise?")
    is_escalated: bool = DecisionField(description="Does this ticket express extreme customer dissatisfaction or threat of cancellation?")
    urgency: UrgencyLevel = DecisionField(description="What is the operational urgency level to resolve this issue?")
    target_department: Department = DecisionField(description="Which company department should handle this request?")

def main():
    engine = OpenJevEngine()
    
    context = (
        "Subject: URGENT: Production database locked and billing was charged twice!\n"
        "Customer says our API returned 500 errors continuously for 2 hours, disrupting their checkout. "
        "They are furious and demanding immediate refund of the duplicate $4,500 fee or they will terminate their enterprise contract."
    )

    print("=== Input Ticket Context ===")
    print(context)
    print("\nEvaluating non-autoregressive decision...\n")

    result = engine.decide(context=context, schema=TicketTriageDecision)

    print("=== Type-Safe Decision Results ===")
    for field_name, val in result.values.items():
        conf_info = result.confidence_scores[field_name]
        print(f"• {field_name}: {val} (Confidence: {conf_info.confidence*100:.1f}%, Entropy: {conf_info.entropy})")
        print(f"  Distribution: {conf_info.probabilities}")

    print(f"\nExecution Latency: {result.latency_ms:.2f} ms")
    
    # Cast directly into verified Pydantic model
    typed_obj = result.get_typed_instance(TicketTriageDecision)
    print(f"Validated Model Object: {typed_obj}")

if __name__ == "__main__":
    main()
