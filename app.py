"""
ChurnGuard AI - Flask Web Application & REST API
Universal platform for customer churn prediction, dataset ingestion,
health scoring, and retention workflows.
"""

import os
import io
import json
from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
from flask_cors import CORS
from dataset_processor import universal_processor
from ml_engine import engine as baseline_engine

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(BASE_DIR, 'samples')

# Track active actions taken on accounts during this session
executed_playbooks = {}

# Initialize with B2B SaaS sample on startup
try:
    default_sample = os.path.join(SAMPLES_DIR, 'sample_b2b_saas.csv')
    if os.path.exists(default_sample):
        universal_processor.analyze_dataset(default_sample)
        universal_processor.process_and_score()
        print("Initialized with default B2B SaaS dataset.")
except Exception as e:
    print(f"Startup init error: {e}")

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/portfolio')
def portfolio():
    return send_from_directory('static', 'portfolio.html')

@app.route('/api/data', methods=['GET'])
def get_current_data():
    """Returns the current scored dataset and summary KPIs."""
    if universal_processor.scored_df is None:
        return jsonify({'error': 'No dataset processed yet'}), 400
    
    # Merge any executed playbook status
    customers = universal_processor.process_and_score()['customers']
    for c in customers:
        c['executed_actions'] = executed_playbooks.get(c['id'], [])

    return jsonify({
        'summary': universal_processor.metrics_summary,
        'mapping': universal_processor.column_mapping,
        'customers': customers
    })

@app.route('/api/upload', methods=['POST'])
def upload_dataset():
    """
    Accepts ANY CSV dataset uploaded by the user.
    Auto-detects columns, trains/scores the model, and returns results.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read uploaded content
        content = file.read().decode('utf-8', errors='ignore')
        stream = io.StringIO(content)
        
        # Analyze schema
        mapping = universal_processor.analyze_dataset(stream)
        
        # Train & Score
        results = universal_processor.process_and_score(mapping)
        executed_playbooks.clear()

        return jsonify({
            'success': True,
            'filename': file.filename,
            'mapping': mapping,
            'summary': results['summary'],
            'total_customers': len(results['customers'])
        })
    except Exception as e:
        return jsonify({'error': f'Failed to process dataset: {str(e)}'}), 500

@app.route('/api/load-sample', methods=['POST'])
def load_sample():
    """Loads one of the pre-built sample datasets (b2b_saas or telco)."""
    data = request.get_json() or {}
    sample_key = data.get('sample', 'b2b_saas')

    filename = 'sample_b2b_saas.csv' if sample_key == 'b2b_saas' else 'sample_telco_subscriptions.csv'
    file_path = os.path.join(SAMPLES_DIR, filename)

    if not os.path.exists(file_path):
        return jsonify({'error': f'Sample file {filename} not found'}), 404

    try:
        mapping = universal_processor.analyze_dataset(file_path)
        results = universal_processor.process_and_score()
        executed_playbooks.clear()
        return jsonify({
            'success': True,
            'sample_loaded': sample_key,
            'mapping': mapping,
            'summary': results['summary'],
            'total_customers': len(results['customers'])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/account/<account_id>', methods=['GET'])
def get_account_detail(account_id):
    """Returns deep-dive diagnostics for a specific customer."""
    if universal_processor.scored_df is None:
        return jsonify({'error': 'No active dataset'}), 400

    customers = universal_processor.process_and_score()['customers']
    matched = next((c for c in customers if str(c['id']) == str(account_id)), None)

    if not matched:
        return jsonify({'error': 'Customer not found'}), 404

    matched['executed_actions'] = executed_playbooks.get(matched['id'], [])
    return jsonify(matched)

@app.route('/api/playbook/trigger', methods=['POST'])
def trigger_playbook():
    """Records an executed retention playbook action for an account."""
    data = request.get_json() or {}
    customer_id = str(data.get('customer_id'))
    action_title = data.get('action_title', 'Intervention Outreach')

    if not customer_id:
        return jsonify({'error': 'Missing customer_id'}), 400

    if customer_id not in executed_playbooks:
        executed_playbooks[customer_id] = []

    action_record = {
        'action_title': action_title,
        'timestamp': 'Just now',
        'status': 'Executed & Dispatched'
    }
    executed_playbooks[customer_id].append(action_record)

    return jsonify({
        'success': True,
        'message': f"Playbook '{action_title}' dispatched successfully for customer {customer_id}.",
        'executed_actions': executed_playbooks[customer_id]
    })

@app.route('/api/export', methods=['GET'])
def export_scored_data():
    """Exports the current customer list with AI Churn & Health Scores as a downloadable CSV."""
    csv_data = universal_processor.export_scored_csv()
    if not csv_data:
        return jsonify({'error': 'No scored data available'}), 400

    return send_file(
        io.BytesIO(csv_data.encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='churnguard_scored_customers.csv'
    )

@app.route('/api/simulator/predict', methods=['POST'])
def simulate_prediction():
    """Live 'What-If' churn simulator prediction endpoint."""
    data = request.get_json() or {}
    result = baseline_engine.predict_account(data)
    return jsonify(result)

if __name__ == '__main__':
    print("Starting ChurnGuard AI Platform on http://127.0.0.1:5000 ...")
    app.run(host='127.0.0.1', port=5000, debug=False)
