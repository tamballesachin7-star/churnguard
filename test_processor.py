import os
from dataset_processor import universal_processor

sample_path = os.path.join(os.path.dirname(__file__), 'samples', 'sample_b2b_saas.csv')
mapping = universal_processor.analyze_dataset(sample_path)
print("Detected Mapping:")
for k, v in mapping.items():
    print(f"  {k}: {v}")

scored = universal_processor.process_and_score()
print("\nScored Summary:")
for k, v in scored['summary'].items():
    print(f"  {k}: {v}")

print("\nSample Scored Customer:")
first_cust = scored['customers'][0]
print(f"  Name: {first_cust['name']}")
print(f"  Risk Level: {first_cust['risk_level']} (Churn Prob: {first_cust['churn_probability']}%, Health: {first_cust['health_score']})")
print(f"  Risk Reasons: {first_cust['risk_reasons']}")
print(f"  Playbooks: {first_cust['playbooks']}")

# Test Telco sample as well!
telco_path = os.path.join(os.path.dirname(__file__), 'samples', 'sample_telco_subscriptions.csv')
telco_mapping = universal_processor.analyze_dataset(telco_path)
print("\n--- Telco Dataset Mapping ---")
for k, v in telco_mapping.items():
    print(f"  {k}: {v}")
telco_scored = universal_processor.process_and_score()
print(f"Telco Accounts Scored: {len(telco_scored['customers'])}, AUC: {telco_scored['summary']['model_auc']}, Accuracy: {telco_scored['summary']['model_accuracy']}%")
