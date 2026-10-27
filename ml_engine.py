"""
ChurnGuard AI - Machine Learning Engine
Predicts B2B customer churn risk, computes customer health scores,
and generates actionable explainability reasons and retention playbooks.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class ChurnPredictionEngine:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = [
            'seat_utilization',         # % active seats / licensed seats (0 - 100)
            'days_since_last_login',    # Recency of login (0 - 45)
            'activity_decay_30d',       # % drop in events compared to baseline (-100 to +50)
            'contract_days_left',       # Days to renewal (0 - 365)
            'open_support_tickets',     # Unresolved issues (0 - 10)
            'csat_score',               # 1 (terrible) to 5 (excellent)
            'exec_sponsor_engaged',     # 1 if executive stakeholder is active, 0 otherwise
            'feature_breadth_score'     # % of available product modules used (0 - 100)
        ]
        self.is_trained = False
        self._train_baseline_model()

    def _generate_training_data(self, n_samples=3000):
        """Generates realistic historical B2B SaaS customer behavior patterns."""
        np.random.seed(42)
        
        seat_util = np.random.beta(a=4, b=2, size=n_samples) * 100
        days_inactive = np.random.exponential(scale=6, size=n_samples).clip(0, 45)
        activity_decay = np.random.normal(loc=-10, scale=35, size=n_samples).clip(-100, 80)
        contract_days = np.random.uniform(5, 365, size=n_samples)
        open_tickets = np.random.poisson(lam=1.2, size=n_samples).clip(0, 8)
        csat = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.08, 0.12, 0.25, 0.35, 0.20])
        exec_sponsor = np.random.binomial(n=1, p=0.65, size=n_samples)
        feature_breadth = np.random.uniform(15, 95, size=n_samples)

        # Ground truth risk calculation with real-world SaaS churn mechanics:
        # High churn probability when:
        # 1. Low seat utilization (<40%)
        # 2. Prolonged inactivity (>14 days)
        # 3. Severe activity decay (< -40%)
        # 4. Approaching contract renewal (< 60 days) combined with dissatisfaction
        # 5. Executive champion left/disengaged
        risk_logit = (
            - 0.05 * (seat_util - 50)
            + 0.15 * days_inactive
            - 0.04 * activity_decay
            - 0.015 * (contract_days - 120)
            + 0.55 * open_tickets
            - 0.70 * (csat - 3)
            - 1.20 * exec_sponsor
            - 0.03 * (feature_breadth - 50)
            - 1.8
        )
        
        churn_prob = 1 / (1 + np.exp(-risk_logit))
        churned = (np.random.uniform(0, 1, size=n_samples) < churn_prob).astype(int)

        df = pd.DataFrame({
            'seat_utilization': seat_util,
            'days_since_last_login': days_inactive,
            'activity_decay_30d': activity_decay,
            'contract_days_left': contract_days,
            'open_support_tickets': open_tickets,
            'csat_score': csat,
            'exec_sponsor_engaged': exec_sponsor,
            'feature_breadth_score': feature_breadth,
            'churned': churned
        })
        return df

    def _train_baseline_model(self):
        """Trains the Random Forest model on historical distribution."""
        df = self._generate_training_data()
        X = df[self.feature_names]
        y = df['churned']

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True

    def predict_account(self, metrics: dict):
        """
        Takes raw account metrics dictionary and outputs:
        - churn_probability (0-100%)
        - health_score (0-100)
        - risk_level ('Healthy', 'At Risk', 'Critical')
        - key_risk_factors (list of actionable explanations)
        - recommended_playbooks (custom retention actions)
        """
        features_vector = [
            float(metrics.get('seat_utilization', 70)),
            float(metrics.get('days_since_last_login', 2)),
            float(metrics.get('activity_decay_30d', 0)),
            float(metrics.get('contract_days_left', 180)),
            float(metrics.get('open_support_tickets', 0)),
            float(metrics.get('csat_score', 4)),
            float(metrics.get('exec_sponsor_engaged', 1)),
            float(metrics.get('feature_breadth_score', 60))
        ]

        scaled_vec = self.scaler.transform([features_vector])
        prob_churn = float(self.model.predict_proba(scaled_vec)[0][1])
        churn_pct = round(prob_churn * 100, 1)

        # Health score is inverse of churn risk with smoothing
        health_score = max(0, min(100, int(100 - churn_pct)))

        if churn_pct >= 60:
            risk_level = "Critical"
        elif churn_pct >= 30:
            risk_level = "At Risk"
        else:
            risk_level = "Healthy"

        # Explainability & Risk Driver Analysis
        risk_factors = []
        if metrics.get('seat_utilization', 70) < 45:
            risk_factors.append({
                'category': 'License Adoption',
                'severity': 'high',
                'description': f"Seat utilization is only {metrics.get('seat_utilization')}%. Over 50% of purchased licenses are dormant."
            })
        if metrics.get('days_since_last_login', 2) >= 10:
            risk_factors.append({
                'category': 'Engagement Inactivity',
                'severity': 'high',
                'description': f"Team has been completely inactive for {int(metrics.get('days_since_last_login'))} consecutive days."
            })
        if metrics.get('activity_decay_30d', 0) <= -35:
            risk_factors.append({
                'category': 'Usage Velocity Drop',
                'severity': 'medium',
                'description': f"Weekly active events dropped by {abs(int(metrics.get('activity_decay_30d')))}% compared to 30-day baseline."
            })
        if metrics.get('contract_days_left', 180) <= 45 and churn_pct > 30:
            risk_factors.append({
                'category': 'Renewal Horizon',
                'severity': 'high',
                'description': f"Contract renews in {int(metrics.get('contract_days_left'))} days while account engagement is declining."
            })
        if metrics.get('open_support_tickets', 0) >= 2:
            risk_factors.append({
                'category': 'Support Friction',
                'severity': 'medium',
                'description': f"{int(metrics.get('open_support_tickets'))} unresolved support tickets causing onboarding/feature friction."
            })
        if metrics.get('exec_sponsor_engaged', 1) == 0:
            risk_factors.append({
                'category': 'Stakeholder Loss',
                'severity': 'high',
                'description': "Executive Champion is disengaged or has left the customer organization."
            })
        if metrics.get('csat_score', 4) <= 2:
            risk_factors.append({
                'category': 'Satisfaction Deficit',
                'severity': 'high',
                'description': f"Recent CSAT rating is low ({metrics.get('csat_score')}/5). Customer has reported dissatisfaction."
            })

        if not risk_factors:
            risk_factors.append({
                'category': 'Steady State',
                'severity': 'low',
                'description': "All usage, adoption, and relationship indicators are within healthy benchmark parameters."
            })

        # Recommended Retention Playbooks based on root cause
        playbooks = []
        if metrics.get('seat_utilization', 70) < 45:
            playbooks.append({
                'id': 'pb_seat_adoption',
                'name': 'Team Onboarding & Seat Activation Workshop',
                'action_type': 'CSM_Intervention',
                'summary': 'Schedule hands-on training with departmental leads to activate dormant seats.',
                'expected_impact': '+25% Seat Utilization'
            })
        if metrics.get('days_since_last_login', 2) >= 10 or metrics.get('activity_decay_30d', 0) <= -35:
            playbooks.append({
                'id': 'pb_reengage_sponsor',
                'name': 'Executive Sponsor Strategic QBR',
                'action_type': 'Executive_Outreach',
                'summary': 'Trigger automated calendar invite from VP of Customer Success for value-realization review.',
                'expected_impact': '-40% Immediate Churn Risk'
            })
        if metrics.get('contract_days_left', 180) <= 60 and churn_pct > 25:
            playbooks.append({
                'id': 'pb_renewal_incentive',
                'name': 'Multi-Year Contract Lock-in & Incentive',
                'action_type': 'Commercial_Offer',
                'summary': 'Offer 15% discount for 2-year commitment or add-on free premium module credits.',
                'expected_impact': 'Guaranteed 24-Month Retention'
            })
        if metrics.get('open_support_tickets', 0) >= 1 or metrics.get('csat_score', 4) <= 2:
            playbooks.append({
                'id': 'pb_escalate_support',
                'name': 'VIP Support Escalation & Ticket Resolution Sprint',
                'action_type': 'Engineering_Escalation',
                'summary': 'Route open tickets directly to Tier-3 engineering lead with 24-hour SLA.',
                'expected_impact': 'Resolves Product Friction'
            })
        
        if not playbooks:
            playbooks.append({
                'id': 'pb_advocate_nurture',
                'name': 'Customer Advocacy & Upsell Nurture',
                'action_type': 'Growth_Campaign',
                'summary': 'Invite customer to join Customer Advisory Board and present an enterprise case study.',
                'expected_impact': 'Expansion Opportunity'
            })

        return {
            'churn_probability': churn_pct,
            'health_score': health_score,
            'risk_level': risk_level,
            'key_risk_factors': risk_factors,
            'recommended_playbooks': playbooks
        }

# Global singleton instance
engine = ChurnPredictionEngine()
