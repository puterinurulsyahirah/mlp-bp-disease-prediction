import os
import threading
import json
import csv
import io
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from model import DiseasePredictionMLP, SYMPTOMS, DISEASES

# Serve compiled React frontend from ../frontend/dist
FRONTEND_DIST = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'dist')
app = Flask(__name__, static_folder=FRONTEND_DIST, static_url_path='/')
# Enable CORS for API routes
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Global manager instance
mlp_manager = DiseasePredictionMLP()
training_thread = None
prediction_history = []

# Load dataset on startup
try:
    mlp_manager.load_data()
except Exception as e:
    print(f"Error loading dataset: {e}")

@app.route('/dataset', methods=['GET'])
def get_dataset():
    try:
        stats = mlp_manager.get_dataset_stats()
        preview = mlp_manager.get_raw_preview(20)
        return jsonify({
            'success': True,
            'dataset_name': 'Disease_Prediction_v1',
            'source': 'Public Health Archive (Kaggle)',
            'updated': 'Oct 2023',
            'rows': stats['rows'],
            'columns': stats['columns'],
            'features_count': stats['features'],
            'diseases_count': stats['diseases'],
            'missing_values': stats['missing_values'],
            'duplicates': stats['duplicates'],
            'preview': preview,
            'disease_dist': stats['disease_dist'],
            'symptom_dist': stats['symptom_dist']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/statistics', methods=['GET'])
def get_statistics():
    try:
        stats = mlp_manager.get_dataset_stats()
        corr_data = mlp_manager.get_correlation_matrix()
        
        # Features summary
        features_summary = []
        for index, item in enumerate(stats['symptom_dist_all']):
            features_summary.append({
                'id': index + 1,
                'name': item['name'],
                'data_type': 'Binary (Integer)',
                'occurrences': item['count'],
                'percentage': item['percentage']
            })
            
        return jsonify({
            'success': True,
            'features_summary': features_summary,
            'data_types': {
                'symptoms': 'Binary (0 or 1)',
                'target': 'Categorical (String)'
            },
            'unique_diseases': stats['diseases'],
            'unique_symptoms': stats['features'],
            'missing_values': stats['missing_values'],
            'duplicate_records': stats['duplicates'],
            'disease_dist': stats['disease_dist'],
            'symptom_dist': stats['symptom_dist'],
            'correlation_heatmap': corr_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/preprocess', methods=['POST'])
def preprocess_dataset():
    try:
        shapes = mlp_manager.preprocess()
        return jsonify({
            'success': True,
            'status': 'completed',
            'original_shape': shapes['original_shape'],
            'processed_shape': shapes['processed_shape'],
            'train_samples': shapes['train_samples'],
            'test_samples': shapes['test_samples'],
            'split_ratio': {
                'training': 70,
                'testing': 30
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def run_training_in_background(params):
    mlp_manager.train_step_by_step(
        hidden_layers=params.get('hidden_layers', (16, 16)),
        activation=params.get('activation', 'relu'),
        optimizer=params.get('optimizer', 'adam'),
        learning_rate=params.get('learning_rate', 0.001),
        epochs=params.get('epochs', 100),
        batch_size=params.get('batch_size', 32),
        random_seed=params.get('random_seed', 42)
    )

@app.route('/train', methods=['POST'])
def train_model():
    global training_thread
    try:
        # Check if already training
        if mlp_manager.training_status['status'] == 'training':
            return jsonify({
                'success': False,
                'message': 'Training is already in progress.'
            }), 400
            
        data = request.json or {}
        
        # Parse hyperparameters
        hidden_input = data.get('hidden_layers', '16,16')
        try:
            hidden_layers = tuple(int(x.strip()) for x in hidden_input.split(','))
        except Exception:
            hidden_layers = (16, 16)
            
        params = {
            'hidden_layers': hidden_layers,
            'activation': data.get('activation', 'relu').lower(),
            'optimizer': data.get('optimizer', 'adam').lower(),
            'learning_rate': float(data.get('learning_rate', 0.001)),
            'epochs': int(data.get('epochs', 100)),
            'batch_size': int(data.get('batch_size', 32)),
            'random_seed': int(data.get('random_seed', 42))
        }
        
        # Reset and start training thread
        mlp_manager.training_status['status'] = 'training'
        training_thread = threading.Thread(target=run_training_in_background, args=(params,))
        training_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Training initiated successfully.',
            'parameters': {
                'hidden_layers': list(hidden_layers),
                'activation': params['activation'],
                'optimizer': params['optimizer'],
                'learning_rate': params['learning_rate'],
                'epochs': params['epochs'],
                'batch_size': params['batch_size'],
                'random_seed': params['random_seed']
            }
        })
    except Exception as e:
        mlp_manager.training_status['status'] = 'idle'
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/training/status', methods=['GET'])
def get_training_status():
    try:
        # Return deep copy or direct status
        return jsonify({
            'success': True,
            'status': mlp_manager.training_status['status'],
            'current_epoch': mlp_manager.training_status['current_epoch'],
            'total_epochs': mlp_manager.training_status['total_epochs'],
            'elapsed_time': mlp_manager.training_status['elapsed_time'],
            'progress': mlp_manager.training_status['progress'],
            'logs': mlp_manager.training_status['logs'],
            'history': mlp_manager.training_status['history']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/evaluation', methods=['GET'])
def get_evaluation():
    try:
        eval_metrics = mlp_manager.get_evaluation_metrics()
        if 'error' in eval_metrics:
            return jsonify({
                'success': False,
                'trained': False,
                'message': eval_metrics['error']
            })
            
        return jsonify({
            'success': True,
            'trained': True,
            'accuracy': eval_metrics['accuracy'],
            'precision': eval_metrics['precision'],
            'recall': eval_metrics['recall'],
            'f1_score': eval_metrics['f1_score'],
            'specificity': eval_metrics['specificity'],
            'confusion_matrix': eval_metrics['confusion_matrix'],
            'classification_report': eval_metrics['classification_report']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_disease():
    try:
        data = request.json or {}
        symptoms = data.get('symptoms', [])
        
        if not symptoms:
            return jsonify({
                'success': False,
                'message': 'No symptoms selected.'
            }), 400
            
        prediction = mlp_manager.predict(symptoms)
        if 'error' in prediction:
            return jsonify({
                'success': False,
                'message': prediction['error']
            }), 400
            
        # Save to prediction history
        import time
        record = {
            'id': f"#PR{str(len(prediction_history)+1).zfill(3)}",
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'symptoms': [s.replace('_', ' ').title() for s in symptoms],
            'prediction': prediction['prognosis'],
            'confidence': prediction['confidence']
        }
        prediction_history.insert(0, record) # Prepend for recent history
        
        return jsonify({
            'success': True,
            'prognosis': prediction['prognosis'],
            'confidence': prediction['confidence'],
            'top_predictions': prediction['top_predictions'],
            'explanations': prediction['explanations'],
            'history_count': len(prediction_history)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/prediction/history', methods=['GET'])
def get_prediction_history():
    return jsonify({
        'success': True,
        'history': prediction_history
    })

@app.route('/reports', methods=['GET'])
def get_reports_info():
    return jsonify({
        'success': True,
        'reports': [
            {
                'id': 'final_report',
                'name': 'Final Project Report',
                'type': 'CSV',
                'description': 'Complete dataset characteristics, selected parameters, and convergence metrics.',
                'download_url': '/reports/download/final_report'
            },
            {
                'id': 'classification_report',
                'name': 'Classification Report',
                'type': 'JSON',
                'description': 'Precision, recall, f1-score, and support metrics for all 41 disease classes.',
                'download_url': '/reports/download/classification_report'
            },
            {
                'id': 'training_logs',
                'name': 'Model Training Logs',
                'type': 'TXT',
                'description': 'Chronological log detailing epochs loss and validation metrics.',
                'download_url': '/reports/download/training_logs'
            },
            {
                'id': 'model_parameters',
                'name': 'MLP Parameters Config',
                'type': 'JSON',
                'description': 'Hyperparameters, weights sizing, and layer specifications.',
                'download_url': '/reports/download/model_parameters'
            },
            {
                'id': 'prediction_history',
                'name': 'Patient Prediction History',
                'type': 'CSV',
                'description': 'History of observed symptoms and model predictions made in this session.',
                'download_url': '/reports/download/prediction_history'
            },
            {
                'id': 'user_manual',
                'name': 'System User Manual',
                'type': 'PDF',
                'description': 'Comprehensive documentation containing user instructions and architecture diagrams.',
                'download_url': '/reports/download/user_manual'
            }
        ]
    })

@app.route('/reports/download/<report_id>', methods=['GET'])
def download_report(report_id):
    try:
        if report_id == 'final_report':
            # Create a CSV summary report
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['Metric / Parameter', 'Value'])
            
            stats = mlp_manager.get_dataset_stats()
            writer.writerow(['Dataset Rows', stats['rows']])
            writer.writerow(['Dataset Features', stats['features']])
            writer.writerow(['Unique Diseases', stats['diseases']])
            
            if mlp_manager.model is not None:
                eval_metrics = mlp_manager.get_evaluation_metrics()
                writer.writerow(['Model Status', 'Healthy (Trained)'])
                writer.writerow(['Training Accuracy', f"{mlp_manager.training_status['history']['train_acc'][-1]:.2f}%" if mlp_manager.training_status['history']['train_acc'] else 'N/A'])
                writer.writerow(['Testing Accuracy', f"{eval_metrics.get('accuracy', 0)*100:.2f}%"])
                writer.writerow(['Precision (Weighted)', eval_metrics.get('precision', 0)])
                writer.writerow(['Recall (Weighted)', eval_metrics.get('recall', 0)])
                writer.writerow(['F1-Score (Weighted)', eval_metrics.get('f1_score', 0)])
            else:
                writer.writerow(['Model Status', 'Untrained'])
                
            mem = io.BytesIO()
            mem.write(output.getvalue().encode('utf-8'))
            mem.seek(0)
            return send_file(
                mem,
                as_attachment=True,
                download_name='final_project_report.csv',
                mimetype='text/csv'
            )
            
        elif report_id == 'classification_report':
            eval_metrics = mlp_manager.get_evaluation_metrics()
            if 'error' in eval_metrics:
                return jsonify(eval_metrics), 400
            
            report_data = eval_metrics['classification_report']
            mem = io.BytesIO()
            mem.write(json.dumps(report_data, indent=4).encode('utf-8'))
            mem.seek(0)
            return send_file(
                mem,
                as_attachment=True,
                download_name='classification_report.json',
                mimetype='application/json'
            )
            
        elif report_id == 'training_logs':
            logs_txt = "\n".join(mlp_manager.training_status['logs'])
            mem = io.BytesIO()
            mem.write(logs_txt.encode('utf-8'))
            mem.seek(0)
            return send_file(
                mem,
                as_attachment=True,
                download_name='model_training_logs.txt',
                mimetype='text/plain'
            )
            
        elif report_id == 'model_parameters':
            params = {}
            if mlp_manager.model is not None:
                params = {
                    'hidden_layer_sizes': mlp_manager.model.hidden_layer_sizes,
                    'activation': mlp_manager.model.activation,
                    'solver': mlp_manager.model.solver,
                    'learning_rate_init': mlp_manager.model.learning_rate_init,
                    'batch_size': mlp_manager.model.batch_size,
                    'random_state': mlp_manager.model.random_state,
                    'classes_count': len(mlp_manager.model.classes_),
                    'n_features_in_': mlp_manager.model.n_features_in_
                }
            else:
                params = {'status': 'Model not trained.'}
                
            mem = io.BytesIO()
            mem.write(json.dumps(params, indent=4).encode('utf-8'))
            mem.seek(0)
            return send_file(
                mem,
                as_attachment=True,
                download_name='mlp_parameters.json',
                mimetype='application/json'
            )
            
        elif report_id == 'prediction_history':
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['ID', 'Timestamp', 'Observed Symptoms', 'Predicted Disease', 'Confidence (%)'])
            for r in prediction_history:
                writer.writerow([r['id'], r['timestamp'], ", ".join(r['symptoms']), r['prediction'], r['confidence']])
                
            mem = io.BytesIO()
            mem.write(output.getvalue().encode('utf-8'))
            mem.seek(0)
            return send_file(
                mem,
                as_attachment=True,
                download_name='prediction_history.csv',
                mimetype='text/csv'
            )
            
        elif report_id == 'user_manual':
            # Generate a text-based documentation manual and download as text/pdf-like representation
            manual_text = """============================================================
MLP DISEASE PREDICTION SYSTEM - USER MANUAL
============================================================
This document explains the system design, dataset schema, model configuration,
training workflow, model evaluation, and prediction inference.

1. APPLICATION WORKFLOW
The computational intelligence workflow operates in six core phases:
   - Load Dataset: Fetches the clinical record corpus (4,920 records, 132 symptoms).
   - Data Preprocessing: Encodes text prognosis targets and scales features.
   - Model Configuration: Users fine-tune hidden layers, learning rates, epochs.
   - Training Process: Initiates Backpropagation on network layers.
   - Model Evaluation: Plots confusion matrix and prints classification f1 scores.
   - Disease Inference: Predicts prognosis categories dynamically.

2. MLP BACKPROPAGATION THEORY
The system utilizes a Multi-Layer Perceptron (MLP) Classifier:
   - Forward Pass: Inputs travel through dense synapses, multiplying weights.
   - Activation: ReLU computes non-linear thresholds in hidden layers.
   - Backpropagation: Calculates gradients of output loss using Adam/SGD.
   - Weight Updating: Updates synaptic strengths dynamically.

3. FAQ & TROUBLESHOOTING
   - Why is accuracy low? Increase epochs or select more representative hidden nodes.
   - Can I input multiple symptoms? Yes, check all observed symptom items.
============================================================
"""
            mem = io.BytesIO()
            mem.write(manual_text.encode('utf-8'))
            mem.seek(0)
            return send_file(
                mem,
                as_attachment=True,
                download_name='disease_prediction_user_manual.txt',
                mimetype='text/plain'
            )
            
        else:
            return jsonify({'success': False, 'message': 'Unknown report type.'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Catch-all route: serve React index.html for all non-API routes
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    full_path = os.path.join(app.static_folder, path)
    if path and os.path.exists(full_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug)
