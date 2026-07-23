"""
Maps fraud_oracle.csv onto data/raw/claims.csv using ONLY real columns
from the source dataset -- nothing synthesized.

What changed from the first version:
- No claim_amount / policy_premium -- those don't exist in the source
  data, so they've been dropped rather than invented. VehiclePrice is
  kept as its real categorical bucket.
- No exact incident_date / claim_filed_date -- only Year + Month exist
  in the source, so we keep a coarse `incident_period` (e.g. "1994-Dec")
  instead of fabricating a day.
- FraudFound_P is kept as the real ground-truth label `fraud_reported`.

Usage:
    python preprocess_dataset.py --input data/raw/fraud_oracle.csv --output data/raw/claims.csv
"""
import argparse

import pandas as pd


def main(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)

    out = pd.DataFrame()
    out["claim_id"] = df["PolicyNumber"].astype(str)
    out["customer_id"] = df["PolicyNumber"].astype(str)
    out["make"] = df["Make"]
    out["accident_area"] = df["AccidentArea"]
    out["sex"] = df["Sex"]
    out["marital_status"] = df["MaritalStatus"]
    out["age"] = df["Age"]
    out["fault"] = df["Fault"]
    out["policy_type"] = df["PolicyType"]
    out["vehicle_category"] = df["VehicleCategory"]
    out["vehicle_price_bucket"] = df["VehiclePrice"]
    out["deductible"] = df["Deductible"]
    out["driver_rating"] = df["DriverRating"]
    out["days_policy_accident"] = df["Days_Policy_Accident"]
    out["days_policy_claim"] = df["Days_Policy_Claim"]
    out["past_number_of_claims"] = df["PastNumberOfClaims"]
    out["age_of_vehicle"] = df["AgeOfVehicle"]
    out["age_of_policyholder"] = df["AgeOfPolicyHolder"]
    out["police_report_filed"] = df["PoliceReportFiled"]
    out["witness_present"] = df["WitnessPresent"]
    out["agent_type"] = df["AgentType"]
    out["number_of_suppliments"] = df["NumberOfSuppliments"]
    out["address_change_claim"] = df["AddressChange_Claim"]
    out["number_of_cars"] = df["NumberOfCars"]
    out["base_policy"] = df["BasePolicy"]
    out["incident_period"] = df["Year"].astype(str) + "-" + df["Month"].astype(str)
    out["fraud_reported"] = df["FraudFound_P"]

    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out)} claims to {output_path} (real columns only, nothing synthesized)")
    print(out.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/fraud_oracle.csv")
    parser.add_argument("--output", default="data/raw/claims.csv")
    args = parser.parse_args()
    main(args.input, args.output)
