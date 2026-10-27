"""
ChurnGuard AI - Realistic Sample Datasets Generator
Creates pre-built datasets for B2B SaaS, Telco, and FinTech to test universal ingestion.
"""

import os
import numpy as np
import pandas as pd

def generate_sample_datasets():
    samples_dir = os.path.join(os.path.dirname(__file__), 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    # 1. B2B SaaS Cloud Platform Dataset (120 accounts)
    np.random.seed(101)
    companies = [
        "Acme Enterprise", "Apex Logistics", "Beacon Analytics", "CloudPulse Systems",
        "DataSphere Corp", "Echo Health", "Falcon Security", "Gridline Networks",
        "Horizon AI", "InfiniStream", "Jupiter Fintech", "Krypton Labs",
        "Luminary Media", "Matrix Dynamics", "Nexus Cloud", "OmniGlobal",
        "Paradigm Shift", "Quantum Leap", "Redwood Retail", "Starlight Software",
        "Titan Technologies", "Unified Commerce", "Vanguard Financial", "Wireframe Studios",
        "Zenith BioTech", "Aegis Defence", "BlueShift Labs", "CipherCore",
        "Delta Solutions", "Element Logic", "Frontier Data", "Genix Digital",
        "Helios Solar", "IronClad Systems", "JetStream Logistics", "Keystone Retail",
        "Lighthouse Media", "Meridian Maritime", "Nova Robotics", "Orbit Space",
        "Pinnacle Advisory", "Quasar Energy", "Raptor Telecom", "Strata Engineering",
        "Terraform Systems", "Ultralight SaaS", "Velox Fleet", "Waveform Studios",
        "Xenon Automation", "YieldFarm AI"
    ]
    # expand to 100+
    b2b_names = [f"{c} {'Inc' if i%2==0 else 'LLC'}" for i, c in enumerate(companies * 2)]
    
    n_b2b = len(b2b_names)
    seat_util = np.random.beta(a=3, b=2, size=n_b2b) * 100
    days_inactive = np.random.exponential(scale=5, size=n_b2b).clip(0, 40).astype(int)
    activity_drop = np.random.normal(loc=-15, scale=30, size=n_b2b).clip(-95, 60).astype(int)
    contract_days = np.random.randint(10, 365, size=n_b2b)
    open_tickets = np.random.poisson(lam=1.1, size=n_b2b)
    csat = np.random.choice([1, 2, 3, 4, 5], size=n_b2b, p=[0.08, 0.12, 0.20, 0.38, 0.22])
    mrr = np.random.choice([1200, 2400, 4800, 8500, 15000, 25000], size=n_b2b, p=[0.3, 0.3, 0.2, 0.1, 0.07, 0.03])
    plan = np.random.choice(["Starter", "Professional", "Enterprise"], size=n_b2b, p=[0.4, 0.4, 0.2])

    # Churn label
    churn_prob = (
        (seat_util < 35).astype(int) * 0.45 +
        (days_inactive > 12).astype(int) * 0.35 +
        (activity_drop < -40).astype(int) * 0.30 +
        (open_tickets >= 3).astype(int) * 0.25 -
        (csat >= 4).astype(int) * 0.35
    ).clip(0, 0.95)
    churn = np.where(np.random.uniform(0, 1, size=n_b2b) < churn_prob, "Yes", "No")

    df_b2b = pd.DataFrame({
        'Account_Name': b2b_names,
        'Plan_Tier': plan,
        'Monthly_Recurring_Revenue': mrr,
        'Seat_Utilization_Pct': np.round(seat_util, 1),
        'Days_Since_Last_Login': days_inactive,
        'Activity_Decay_30d_Pct': activity_drop,
        'Days_Until_Renewal': contract_days,
        'Open_Support_Tickets': open_tickets,
        'CSAT_Rating': csat,
        'Churn': churn
    })
    b2b_path = os.path.join(samples_dir, 'sample_b2b_saas.csv')
    df_b2b.to_csv(b2b_path, index=False)

    # 2. Telco / Consumer Subscriptions Dataset
    np.random.seed(202)
    n_telco = 150
    customer_ids = [f"TELCO-{1000 + i}" for i in range(n_telco)]
    tenure = np.random.randint(1, 72, size=n_telco)
    monthly_charges = np.random.uniform(25.0, 118.0, size=n_telco).round(2)
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_telco, p=[0.55, 0.25, 0.20])
    tech_support = np.random.choice(["Yes", "No"], size=n_telco, p=[0.4, 0.6])
    paperless_billing = np.random.choice(["Yes", "No"], size=n_telco, p=[0.6, 0.4])
    payment_method = np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n_telco)
    
    t_churn_prob = (
        (contract == "Month-to-month").astype(int) * 0.40 +
        (monthly_charges > 80).astype(int) * 0.25 +
        (tech_support == "No").astype(int) * 0.20 -
        (tenure > 36).astype(int) * 0.35
    ).clip(0.05, 0.90)
    t_churn = np.where(np.random.uniform(0, 1, size=n_telco) < t_churn_prob, "Yes", "No")

    df_telco = pd.DataFrame({
        'CustomerID': customer_ids,
        'Tenure_Months': tenure,
        'Contract': contract,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': (monthly_charges * tenure).round(2),
        'TechSupport': tech_support,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'Churn': t_churn
    })
    telco_path = os.path.join(samples_dir, 'sample_telco_subscriptions.csv')
    df_telco.to_csv(telco_path, index=False)

    print(f"Generated {b2b_path} ({len(df_b2b)} rows)")
    print(f"Generated {telco_path} ({len(df_telco)} rows)")

if __name__ == '__main__':
    generate_sample_datasets()
