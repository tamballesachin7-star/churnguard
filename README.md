# 🛡️ ChurnGuard AI

**Universal Customer Churn Prediction & Retention Intelligence Platform**

ChurnGuard AI predicts customer churn *before* it happens. By ingesting behavioral telemetry, usage decay, and subscription health signals, it computes a real-time **Customer Health Score (0–100)**, identifies **Primary Churn Drivers**, and recommends **Targeted Retention Playbooks**.

---

## 🌟 Key Capabilities

1. **Universal Dataset Ingestion (Any Company Dataset):**
   - Upload any customer dataset (`.csv`) via drag-and-drop or file picker.
   - **Smart Column Detection:** Automatically infers Account/Customer ID, Churn Labels, Financial Spend/MRR, Inactivity/Recency, Usage Volume, and Support Tickets.
   - **AutoML Pipeline:** Automatically trains a Gradient Boosting / Random Forest classifier tailored to your uploaded data, computing ROC-AUC and Accuracy.

2. **Predictive Churn & Health Scoring:**
   - Multi-factor risk calculation combining behavioral velocity, inactivity, and contract timing.
   - Accounts classified into **Healthy**, **At Risk**, and **Critical**.

3. **Explainability ("Why is this client leaving?"):**
   - Surfaces root causes (e.g., *"Team inactive for 14 consecutive days"*, *"Seat utilization dropped to 28%"*, *"3 unresolved billing tickets"*).

4. **Actionable Retention Playbooks:**
   - Automated workflows (e.g. *"Team Activation Workshop"*, *"Executive Sponsor Outreach"*, *"Commercial Retention Offer"*).
   - Track dispatch status directly from the dashboard.

5. **Live "What-If" Simulator:**
   - Interactive sliders for seat utilization, inactivity days, usage decay, and support tickets with real-time recalculation of churn probability.

6. **Full Scored Data Export:**
   - One-click export of your dataset enriched with `Churn_Probability`, `Health_Score`, and `Risk_Level`.

---

## 🚀 Quick Start

### 1. Launch the Application

Navigate to the project directory and run:

```bash
cd "C:\Users\T14\.gemini\antigravity\scratch\churnguard"
python app.py
```

### 2. Open the Dashboard

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 📂 Project Architecture

```
churnguard/
├── app.py                   # Flask REST API & Web Application
├── dataset_processor.py     # Universal dataset ingestion, schema inference & AutoML
├── ml_engine.py             # Baseline churn prediction engine & playbook logic
├── generate_samples.py      # Generates realistic B2B SaaS & Telco test datasets
├── samples/
│   ├── sample_b2b_saas.csv             # 100 B2B SaaS company accounts
│   └── sample_telco_subscriptions.csv  # 150 Consumer/Telco subscribers
├── static/
│   ├── index.html           # Modern glassmorphism Single-Page Application
│   ├── app.js               # Frontend orchestration, charts, filtering & simulator
│   └── style.css            # Custom CSS animations and styles
└── README.md                # System documentation & guide
```

---

## 🔌 REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `GET /api/data` | `GET` | Fetches current dataset summary, metrics, and accounts |
| `POST /api/upload` | `POST` | Uploads and processes any custom company CSV |
| `POST /api/load-sample` | `POST` | Loads pre-built sample (`b2b_saas` or `telco`) |
| `GET /api/account/<id>` | `GET` | Deep dive into an individual customer's metrics & trend |
| `POST /api/playbook/trigger` | `POST` | Dispatches an automated retention playbook |
| `GET /api/export` | `GET` | Downloads the enriched CSV with churn scores |
| `POST /api/simulator/predict`| `POST` | Runs live "What-If" churn simulation |
