import json

SYSTEM_PROMPT = """You are a claims risk reasoning assistant for an insurance adjuster.
You will be given a claim's attributes and three model-generated risk scores
(fraud_probability, anomaly_score, customer_risk_index -- all 0 to 1, higher = riskier).

Respond with ONLY a JSON object, no other text, no markdown code fences, matching exactly:
{
  "discrepancy_detected": true or false,
  "flagged_reasons": ["short reason 1", "short reason 2"],
  "recommended_action": "APPROVE" or "REVIEW" or "REJECT_AND_INVESTIGATE",
  "rationale_text": "a concise 2-sentence explanation an adjuster can read in 5 seconds"
}

Base your reasoning on the given scores and claim attributes only. Be concise and specific."""


def build_prompt(claim: dict) -> str:
    return json.dumps({
        "claim_id": claim["claim_id"],
        "vehicle_category": claim.get("vehicle_category"),
        "fault": claim.get("fault"),
        "base_policy": claim.get("base_policy"),
        "police_report_filed": claim.get("police_report_filed"),
        "witness_present": claim.get("witness_present"),
        "past_number_of_claims": claim.get("past_number_of_claims"),
        "address_change_claim": claim.get("address_change_claim"),
        "deductible": claim.get("deductible"),
        "fraud_probability": round(float(claim["fraud_probability"]), 3),
        "anomaly_score": round(float(claim["anomaly_score"]), 3),
        "customer_risk_index": round(float(claim["customer_risk_index"]), 3),
    }, indent=2)
