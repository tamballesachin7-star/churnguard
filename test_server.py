"""
Test script for ChurnGuard AI using Flask test client.
Tests the web page, dataset loading, universal CSV upload, prediction API, and playbook dispatch.
"""

import io
from app import app

def test_churnguard():
    client = app.test_client()

    # 1. Test Index Web Page
    res = client.get('/')
    assert res.status_code == 200
    assert b"ChurnGuard" in res.data
    print("PASS: Index HTML rendered successfully (Status 200)")

    # 2. Test /api/data
    res = client.get('/api/data')
    assert res.status_code == 200
    data = res.get_json()
    assert 'summary' in data
    assert 'customers' in data
    assert len(data['customers']) > 0
    print(f"PASS: /api/data returned {len(data['customers'])} customers, Total MRR: ${data['summary']['total_mrr']}")

    # 3. Test /api/simulator/predict
    sim_payload = {
        'seat_utilization': 25,
        'days_since_last_login': 18,
        'activity_decay_30d': -60,
        'contract_days_left': 20,
        'open_support_tickets': 3,
        'csat_score': 2,
        'exec_sponsor_engaged': 0
    }
    res = client.post('/api/simulator/predict', json=sim_payload)
    assert res.status_code == 200
    sim_res = res.get_json()
    assert sim_res['risk_level'] == 'Critical'
    assert sim_res['churn_probability'] > 60
    print(f"PASS: Simulator correctly predicted Critical Churn ({sim_res['churn_probability']}%, Health: {sim_res['health_score']})")

    # 4. Test Universal File Upload (/api/upload) with a completely arbitrary company CSV format!
    arbitrary_csv = (
        "ClientOrg,AnnualValue,IdleDaysSinceUsage,SupportIssuesCount,LicenseUsagePct,LeftCompany\n"
        "GlobalTech,120000,2,0,92,No\n"
        "ApexLogistics,84000,1,1,85,No\n"
        "OmegaRetail,45000,22,4,15,Yes\n"
        "SolarisEnergy,95000,18,3,28,Yes\n"
        "DeltaDynamics,60000,3,0,78,No\n"
    )
    upload_data = {
        'file': (io.BytesIO(arbitrary_csv.encode('utf-8')), 'arbitrary_company_dataset.csv')
    }
    res = client.post('/api/upload', data=upload_data, content_type='multipart/form-data')
    assert res.status_code == 200
    upload_res = res.get_json()
    assert upload_res['success'] is True
    print(f"PASS: Universal Upload successfully ingested custom company dataset! Auto-detected mapping: {upload_res['mapping']['id_col']} -> {upload_res['total_customers']} accounts scored.")

    # 5. Test Playbook Trigger
    res = client.post('/api/playbook/trigger', json={'customer_id': 'OmegaRetail', 'action_title': 'Executive Outreach'})
    assert res.status_code == 200
    pb_res = res.get_json()
    assert pb_res['success'] is True
    print(f"PASS: Playbook triggered successfully: {pb_res['message']}")

    # 6. Test CSV Export
    res = client.get('/api/export')
    assert res.status_code == 200
    assert b"_churn_prob" in res.data or b"ClientOrg" in res.data
    print("PASS: Enriched CSV export generated successfully!")

    print("\nALL 6 AUTOMATED END-TO-END TESTS PASSED WITH 100% SUCCESS!")

if __name__ == '__main__':
    test_churnguard()
