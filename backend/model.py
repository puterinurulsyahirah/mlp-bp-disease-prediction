import os
import urllib.request
import pandas as pd
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# List of 132 symptoms in the standard dataset
SYMPTOMS = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain',
    'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
    'spotting_urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings',
    'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
    'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache',
    'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain',
    'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
    'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
    'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
    'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
    'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus',
    'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels',
    'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger',
    'extra_marital_contacts', 'drying_of_peels_and_skin', 'internal_itching', 'toxic_look_typhos',
    'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
    'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes', 'increased_appetite', 'polyuria',
    'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
    'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
    'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'prominent_veins_on_calf',
    'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
    'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
    'yellow_crust_ooze'
]

# List of 41 diseases
DISEASES = [
    'Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction', 'Peptic ulcer disease',
    'AIDS', 'Diabetes ', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine',
    'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue',
    'Typhoid', 'hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
    'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)',
    'Heart attack', 'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis',
    'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis',
    'Impetigo'
]

# Mapping of diseases to typical symptoms to generate realistic synthetic data
DISEASE_SYMPTOMS_MAP = {
    'Fungal infection': ['itching', 'skin_rash', 'nodal_skin_eruptions', 'dischromic_patches'],
    'Allergy': ['continuous_sneezing', 'shivering', 'chills', 'watering_from_eyes'],
    'GERD': ['stomach_pain', 'acidity', 'ulcers_on_tongue', 'vomiting', 'cough', 'chest_pain'],
    'Chronic cholestasis': ['itching', 'vomiting', 'yellowish_skin', 'nausea', 'loss_of_appetite', 'yellowing_of_eyes'],
    'Drug Reaction': ['itching', 'skin_rash', 'stomach_pain', 'burning_micturition', 'spotting_urination'],
    'Peptic ulcer disease': ['vomiting', 'indigestion', 'loss_of_appetite', 'abdominal_pain'],
    'AIDS': ['muscle_wasting', 'patches_in_throat', 'high_fever', 'extra_marital_contacts'],
    'Diabetes ': ['fatigue', 'weight_loss', 'restlessness', 'lethargy', 'irregular_sugar_level', 'blurred_and_distorted_vision', 'obesity', 'excessive_hunger', 'increased_appetite', 'polyuria'],
    'Gastroenteritis': ['vomiting', 'dehydration', 'diarrhoea'],
    'Bronchial Asthma': ['fatigue', 'cough', 'high_fever', 'breathlessness', 'family_history', 'mucoid_sputum'],
    'Hypertension ': ['headache', 'chest_pain', 'dizziness', 'loss_of_balance', 'lack_of_concentration'],
    'Migraine': ['acidity', 'indigestion', 'headache', 'blurred_and_distorted_vision', 'excessive_hunger', 'stiff_neck', 'depression', 'irritability', 'visual_disturbances'],
    'Cervical spondylosis': ['back_pain', 'neck_pain', 'dizziness', 'loss_of_balance'],
    'Paralysis (brain hemorrhage)': ['vomiting', 'headache', 'weakness_in_limbs', 'altered_sensorium'],
    'Jaundice': ['itching', 'vomiting', 'fatigue', 'weight_loss', 'high_fever', 'yellowish_skin', 'dark_urine', 'abdominal_pain'],
    'Malaria': ['chills', 'vomiting', 'high_fever', 'sweating', 'headache', 'muscle_pain'],
    'Chicken pox': ['itching', 'skin_rash', 'fatigue', 'lethargy', 'high_fever', 'headache', 'loss_of_appetite', 'mild_fever', 'swelled_lymph_nodes', 'malaise', 'red_spots_over_body'],
    'Dengue': ['skin_rash', 'chills', 'joint_pain', 'vomiting', 'fatigue', 'high_fever', 'headache', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'muscle_pain', 'red_spots_over_body'],
    'Typhoid': ['chills', 'vomiting', 'fatigue', 'high_fever', 'headache', 'nausea', 'loss_of_appetite', 'constipation', 'abdominal_pain', 'diarrhoea', 'toxic_look_typhos', 'belly_pain'],
    'hepatitis A': ['joint_pain', 'vomiting', 'mild_fever', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'abdominal_pain', 'diarrhoea', 'yellowing_of_eyes'],
    'Hepatitis B': ['itching', 'fatigue', 'lethargy', 'yellowish_skin', 'dark_urine', 'loss_of_appetite', 'abdominal_pain', 'yellowing_of_eyes', 'receiving_blood_transfusion', 'receiving_unsterile_injections'],
    'Hepatitis C': ['fatigue', 'yellowish_skin', 'nausea', 'loss_of_appetite', 'yellowing_of_eyes', 'family_history'],
    'Hepatitis D': ['joint_pain', 'vomiting', 'fatigue', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'abdominal_pain', 'yellowing_of_eyes'],
    'Hepatitis E': ['joint_pain', 'vomiting', 'fatigue', 'high_fever', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'abdominal_pain', 'yellowing_of_eyes', 'acute_liver_failure', 'coma', 'stomach_bleeding'],
    'Alcoholic hepatitis': ['vomiting', 'yellowish_skin', 'abdominal_pain', 'swelling_of_stomach', 'history_of_alcohol_consumption', 'fluid_overload'],
    'Tuberculosis': ['chills', 'vomiting', 'fatigue', 'weight_loss', 'cough', 'high_fever', 'breathlessness', 'sweating', 'loss_of_appetite', 'mild_fever', 'phlegm', 'chest_pain', 'blood_in_sputum'],
    'Common Cold': ['continuous_sneezing', 'chills', 'fatigue', 'cough', 'high_fever', 'headache', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'muscle_pain'],
    'Pneumonia': ['chills', 'fatigue', 'cough', 'high_fever', 'breathlessness', 'sweating', 'chest_pain', 'fast_heart_rate', 'rusty_sputum'],
    'Dimorphic hemmorhoids(piles)': ['constipation', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus'],
    'Heart attack': ['vomiting', 'breathlessness', 'sweating', 'chest_pain', 'palpitations'],
    'Varicose veins': ['fatigue', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'prominent_veins_on_calf'],
    'Hypothyroidism': ['fatigue', 'weight_gain', 'cold_hands_and_feets', 'mood_swings', 'lethargy', 'dizziness', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'depression', 'irritability'],
    'Hyperthyroidism': ['fatigue', 'weight_loss', 'restlessness', 'sweating', 'diarrhoea', 'fast_heart_rate', 'excessive_hunger', 'muscle_weakness', 'abnormal_menstruation'],
    'Hypoglycemia': ['fatigue', 'anxiety', 'sweating', 'headache', 'nausea', 'blurred_and_distorted_vision', 'excessive_hunger', 'slurred_speech', 'irritability', 'palpitations'],
    'Osteoarthristis': ['joint_pain', 'neck_pain', 'dizziness', 'painful_walking', 'swelling_joints', 'stiff_joints'],
    'Arthritis': ['muscle_weakness', 'stiff_joints', 'movement_stiffness', 'painful_walking', 'swelling_joints'],
    '(vertigo) Paroymsal  Positional Vertigo': ['vomiting', 'headache', 'nausea', 'dizziness', 'loss_of_balance', 'spinning_movements'],
    'Acne': ['skin_rash', 'pus_filled_pimples', 'blackheads', 'scurring'],
    'Urinary tract infection': ['burning_micturition', 'bladder_discomfort', 'foul_smell_of_urine', 'continuous_feel_of_urine'],
    'Psoriasis': ['skin_rash', 'joint_pain', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails'],
    'Impetigo': ['skin_rash', 'high_fever', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']
}

DATASET_FALLBACK_URLS = {
    'Training.csv': [
        'https://raw.githubusercontent.com/itssaurabh22/Disease-Prediction-Using-Machine-Learning/master/Dataset/Training.csv',
        'https://raw.githubusercontent.com/shubham-kumbhar/Disease-Prediction-using-Machine-Learning/master/Dataset/Training.csv',
        'https://raw.githubusercontent.com/ravikiran-bhonagiri/Disease-Prediction-using-Machine-Learning/master/Training.csv',
        'https://raw.githubusercontent.com/kaushikbhupendra/Disease-Prediction-Using-Machine-Learning/master/Dataset/Training.csv'
    ],
    'Testing.csv': [
        'https://raw.githubusercontent.com/itssaurabh22/Disease-Prediction-Using-Machine-Learning/master/Dataset/Testing.csv',
        'https://raw.githubusercontent.com/shubham-kumbhar/Disease-Prediction-using-Machine-Learning/master/Dataset/Testing.csv',
        'https://raw.githubusercontent.com/ravikiran-bhonagiri/Disease-Prediction-using-Machine-Learning/master/Testing.csv',
        'https://raw.githubusercontent.com/kaushikbhupendra/Disease-Prediction-Using-Machine-Learning/master/Dataset/Testing.csv'
    ]
}

class DiseasePredictionMLP:
    def __init__(self, data_dir='data', model_dir='models'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.model_path = os.path.join(self.model_dir, 'mlp_model.joblib')
        self.scaler_path = os.path.join(self.model_dir, 'scaler.joblib')
        self.encoder_path = os.path.join(self.model_dir, 'encoder.joblib')
        
        # Ensure folders exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.df_train = None
        self.df_test = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.is_preprocessed = False
        
        # Status tracker for live training
        self.training_status = {
            'status': 'idle',  # 'idle', 'training', 'completed'
            'current_epoch': 0,
            'total_epochs': 100,
            'elapsed_time': 0,
            'progress': 0,
            'logs': [],
            'history': {
                'epochs': [],
                'loss': [],
                'train_acc': [],
                'test_acc': []
            }
        }
        
    def download_or_generate_dataset(self):
        """Attempts to download dataset from multiple GitHub mirrors. If all fail, generates a synthetic dataset."""
        train_path = os.path.join(self.data_dir, 'Training.csv')
        test_path = os.path.join(self.data_dir, 'Testing.csv')
        
        # Check if already exists
        if os.path.exists(train_path) and os.path.exists(test_path):
            try:
                self.df_train = pd.read_csv(train_path)
                self.df_test = pd.read_csv(test_path)
                return "Loaded existing local dataset files."
            except Exception:
                pass
        
        # Try to download Training
        downloaded = False
        for url in DATASET_FALLBACK_URLS['Training.csv']:
            try:
                urllib.request.urlretrieve(url, train_path)
                # Verify it is valid CSV
                pd.read_csv(train_path)
                downloaded = True
                break
            except Exception:
                if os.path.exists(train_path):
                    os.remove(train_path)
        
        # Try to download Testing
        if downloaded:
            for url in DATASET_FALLBACK_URLS['Testing.csv']:
                try:
                    urllib.request.urlretrieve(url, test_path)
                    pd.read_csv(test_path)
                    break
                except Exception:
                    if os.path.exists(test_path):
                        os.remove(test_path)
        
        if os.path.exists(train_path) and os.path.exists(test_path):
            self.df_train = pd.read_csv(train_path)
            self.df_test = pd.read_csv(test_path)
            return "Successfully downloaded dataset files from GitHub."
            
        # Fallback to generating high-quality synthetic data
        self.generate_synthetic_data()
        return "Offline fallback: Generated high-fidelity synthetic clinical symptoms dataset."
        
    def generate_synthetic_data(self):
        """Generates a synthetic copy of the dataset containing 4920 training samples and 300 test samples."""
        # 120 samples per disease for training
        train_rows = []
        # 10 samples per disease for testing
        test_rows = []
        
        np.random.seed(42)
        
        # Build symptoms schema
        columns = SYMPTOMS.copy()
        
        for disease in DISEASES:
            char_symptoms = DISEASE_SYMPTOMS_MAP.get(disease, [])
            
            # Generate training
            for _ in range(120):
                row = {sym: 0 for sym in columns}
                for sym in columns:
                    if sym in char_symptoms:
                        # Present with 80% probability
                        row[sym] = 1 if np.random.rand() < 0.85 else 0
                    else:
                        # Noise symptom with 1% probability
                        row[sym] = 1 if np.random.rand() < 0.015 else 0
                row['prognosis'] = disease
                train_rows.append(row)
                
            # Generate testing
            for _ in range(10):
                row = {sym: 0 for sym in columns}
                for sym in columns:
                    if sym in char_symptoms:
                        row[sym] = 1 if np.random.rand() < 0.9 else 0
                    else:
                        row[sym] = 1 if np.random.rand() < 0.01 else 0
                row['prognosis'] = disease
                test_rows.append(row)
                
        # Create DataFrames
        self.df_train = pd.DataFrame(train_rows)
        self.df_test = pd.DataFrame(test_rows)
        
        # Write to files
        self.df_train.to_csv(os.path.join(self.data_dir, 'Training.csv'), index=False)
        self.df_test.to_csv(os.path.join(self.data_dir, 'Testing.csv'), index=False)
        
    def load_data(self):
        """Initializes dataset files."""
        return self.download_or_generate_dataset()

    def get_raw_preview(self, count=10):
        """Returns the first 'count' rows of the dataset as a list of dicts for the preview table."""
        if self.df_train is None:
            self.load_data()
        df = self.df_train.copy()
        # Add index column for ID
        df.insert(0, 'ID', [f'#DP{str(i+1).zfill(3)}' for i in range(len(df))])
        return df.head(count).to_dict(orient='records')
        
    def get_dataset_stats(self):
        """Returns key statistical summary of features and target classes."""
        if self.df_train is None:
            self.load_data()
            
        rows, cols = self.df_train.shape
        missing_values = int(self.df_train.isnull().sum().sum())
        # Duplicate rows ignoring prognosis
        duplicates = int(self.df_train.duplicated(subset=SYMPTOMS).sum())
        
        # Calculate Disease distribution
        disease_counts = self.df_train['prognosis'].value_counts()
        disease_dist = []
        total_diseases = len(disease_counts)
        for name, cnt in disease_counts.items():
            disease_dist.append({
                'name': str(name).strip(),
                'count': int(cnt),
                'percentage': float(round((cnt / rows) * 100, 2))
            })
            
        # Calculate top symptom counts
        symptom_sums = self.df_train[SYMPTOMS].sum().sort_values(ascending=False)
        symptom_dist = []
        for name, cnt in symptom_sums.items():
            symptom_dist.append({
                'name': name.replace('_', ' ').title(),
                'count': int(cnt),
                'percentage': float(round((cnt / rows) * 100, 2))
            })
            
        return {
            'rows': rows,
            'columns': cols,
            'features': len(SYMPTOMS),
            'diseases': total_diseases,
            'missing_values': missing_values,
            'duplicates': duplicates,
            'disease_dist': disease_dist,
            'symptom_dist': symptom_dist[:10], # Top 10
            'symptom_dist_all': symptom_dist
        }

    def get_correlation_matrix(self):
        """Returns a subset correlation heatmap of top symptoms."""
        if self.df_train is None:
            self.load_data()
        
        # Get top 15 most frequent symptoms
        symptom_sums = self.df_train[SYMPTOMS].sum().sort_values(ascending=False)
        top_15_symptoms = list(symptom_sums.index[:15])
        
        # Compute correlation
        corr = self.df_train[top_15_symptoms].corr().fillna(0).values
        
        return {
            'features': [f.replace('_', ' ').title() for f in top_15_symptoms],
            'values': corr.tolist()
        }

    def preprocess(self):
        """Processes training and test sets, encodes target class, normalizes features."""
        if self.df_train is None or self.df_test is None:
            self.load_data()
            
        # Remove empty columns if any
        df_tr = self.df_train.dropna(axis=1, how='all')
        df_ts = self.df_test.dropna(axis=1, how='all')
        
        # Separate X and y
        X_tr = df_tr[SYMPTOMS].values.astype(float)
        y_tr = df_tr['prognosis'].values
        
        X_ts = df_ts[SYMPTOMS].values.astype(float)
        y_ts = df_ts['prognosis'].values
        
        # Fit normalizer and label encoder
        self.scaler.fit(X_tr)
        self.label_encoder.fit(y_tr)
        
        # Transform data
        self.X_train = self.scaler.transform(X_tr)
        self.y_train = self.label_encoder.transform(y_tr)
        
        self.X_test = self.scaler.transform(X_ts)
        self.y_test = self.label_encoder.transform(y_ts)
        
        self.is_preprocessed = True
        
        # Save preprocessors
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.label_encoder, self.encoder_path)
        
        return {
            'original_shape': self.df_train.shape,
            'processed_shape': (self.X_train.shape[0], self.X_train.shape[1] + 1),
            'train_samples': self.X_train.shape[0],
            'test_samples': self.X_test.shape[0]
        }

    def train_step_by_step(self, hidden_layers=(16, 16), activation='relu',
                          optimizer='adam', learning_rate=0.001, epochs=100,
                          batch_size=32, random_seed=42):
        """Performs step-by-step training of the MLP model using partial_fit to provide live status updates."""
        if not self.is_preprocessed:
            self.preprocess()
            
        import time
        start_time = time.time()
        
        # Initialize MLPClassifier
        # adam uses partial_fit step by step, we set the initial learning rate
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver=optimizer if optimizer in ['adam', 'sgd'] else 'adam',
            learning_rate_init=learning_rate,
            random_state=random_seed,
            batch_size=batch_size
        )
        
        classes = np.arange(len(self.label_encoder.classes_))
        
        # Setup status
        self.training_status['status'] = 'training'
        self.training_status['total_epochs'] = epochs
        self.training_status['current_epoch'] = 0
        self.training_status['elapsed_time'] = 0
        self.training_status['progress'] = 0
        self.training_status['logs'] = [
            f"[INFO] Initializing MLP architecture: {len(SYMPTOMS)} Inputs -> {hidden_layers} Hidden Nodes -> {len(classes)} Outputs",
            f"[INFO] Training parameters: Optimizer={optimizer}, Learning Rate={learning_rate}, Batch Size={batch_size}, Seed={random_seed}",
            f"[INFO] Preprocessed training samples: {self.X_train.shape[0]}, testing samples: {self.X_test.shape[0]}"
        ]
        self.training_status['history'] = {
            'epochs': [],
            'loss': [],
            'train_acc': [],
            'test_acc': []
        }
        
        n_samples = self.X_train.shape[0]
        
        for epoch in range(1, epochs + 1):
            if self.training_status['status'] != 'training':
                break
                
            # Shuffle inputs
            indices = np.arange(n_samples)
            np.random.seed(random_seed + epoch)
            np.random.shuffle(indices)
            X_shuffled = self.X_train[indices]
            y_shuffled = self.y_train[indices]
            
            # Slices and fit
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                self.model.partial_fit(X_batch, y_batch, classes=classes)
                
            # Metrics
            loss = float(self.model.loss_)
            train_acc = float(self.model.score(self.X_train, self.y_train))
            test_acc = float(self.model.score(self.X_test, self.y_test))
            
            elapsed = round(time.time() - start_time, 2)
            
            # Update status
            self.training_status['current_epoch'] = epoch
            self.training_status['elapsed_time'] = elapsed
            self.training_status['progress'] = int((epoch / epochs) * 100)
            
            # Log progress
            log_str = f"Epoch {epoch:03d}/{epochs:03d} - Loss: {loss:.4f} - Train Acc: {train_acc*100:.2f}% - Test Acc: {test_acc*100:.2f}% (Elapsed: {elapsed}s)"
            self.training_status['logs'].append(log_str)
            
            self.training_status['history']['epochs'].append(epoch)
            self.training_status['history']['loss'].append(loss)
            self.training_status['history']['train_acc'].append(train_acc * 100)
            self.training_status['history']['test_acc'].append(test_acc * 100)
            
            # Small artificial throttle to simulate calculations and support UI updates
            time.sleep(0.02)
            
        # Training complete
        if self.training_status['status'] == 'training':
            self.training_status['status'] = 'completed'
            self.training_status['logs'].append(f"[INFO] Model optimization completed in {self.training_status['elapsed_time']} seconds.")
            self.training_status['logs'].append(f"[INFO] Saving optimized weights & serialized model binary...")
            # Save final model
            joblib.dump(self.model, self.model_path)
            
    def get_evaluation_metrics(self):
        """Computes confusion matrix, classification report, and class-wise precision/recall/F1."""
        if self.model is None:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.label_encoder = joblib.load(self.encoder_path)
                self.is_preprocessed = True
            else:
                return {"error": "Model has not been trained yet."}
                
        # Run test predictions
        y_pred = self.model.predict(self.X_test)
        
        # Overall metrics
        acc = self.model.score(self.X_test, self.y_test)
        
        # Calculate precision, recall, f1
        precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_pred, average='weighted', zero_division=0)
        
        # Class-wise report
        cls_rep = classification_report(self.y_test, y_pred, target_names=self.label_encoder.classes_, output_dict=True, zero_division=0)
        
        # Confusion matrix
        conf_mat = confusion_matrix(self.y_test, y_pred)
        
        # Make a readable JSON confusion matrix
        classes = self.label_encoder.classes_.tolist()
        
        # Performance summary
        # Compute specificity (True Negatives / (True Negatives + False Positives))
        # For multi-class, compute average specificity across classes
        specificities = []
        for i in range(len(classes)):
            temp_y_test = (self.y_test == i).astype(int)
            temp_y_pred = (y_pred == i).astype(int)
            tn, fp, fn, tp = confusion_matrix(temp_y_test, temp_y_pred).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificities.append(spec)
        avg_specificity = float(np.mean(specificities))
        
        return {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'specificity': avg_specificity,
            'confusion_matrix': {
                'classes': classes,
                'matrix': conf_mat.tolist()
            },
            'classification_report': cls_rep
        }

    def predict(self, active_symptoms_names):
        """Predicts disease based on symptom list and generates perturbation feature importance."""
        if self.model is None:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.label_encoder = joblib.load(self.encoder_path)
                self.is_preprocessed = True
            else:
                return {"error": "Model has not been trained yet. Please configure and train it first."}
                
        # Build binary feature vector
        vector = np.zeros(len(SYMPTOMS))
        active_indices = []
        for s in active_symptoms_names:
            if s in SYMPTOMS:
                idx = SYMPTOMS.index(s)
                vector[idx] = 1
                active_indices.append(idx)
                
        # Transform using scaler
        scaled_vector = self.scaler.transform(vector.reshape(1, -1))
        
        # Predict probability
        probs = self.model.predict_proba(scaled_vector)[0]
        
        # Top 3 predicted diseases
        top_indices = np.argsort(probs)[::-1][:3]
        top_predictions = []
        for rank, idx in enumerate(top_indices):
            top_predictions.append({
                'rank': rank + 1,
                'disease': str(self.label_encoder.classes_[idx]).strip(),
                'probability': float(round(probs[idx] * 100, 2))
            })
            
        prognosis = top_predictions[0]['disease']
        confidence = top_predictions[0]['probability']
        prognosis_idx = top_indices[0]
        
        # Perturbation explanation (LIME-style)
        # Compute how much output probability falls if each active symptom is toggled to 0
        explanations = []
        for idx in active_indices:
            perturbed_vector = vector.copy()
            perturbed_vector[idx] = 0 # toggle off
            
            scaled_perturbed = self.scaler.transform(perturbed_vector.reshape(1, -1))
            perturbed_probs = self.model.predict_proba(scaled_perturbed)[0]
            
            # Probability drop for the top predicted disease
            prob_drop = probs[prognosis_idx] - perturbed_probs[prognosis_idx]
            explanations.append({
                'symptom': SYMPTOMS[idx].replace('_', ' ').title(),
                'importance': float(prob_drop),
                'percentage_drop': float(round(prob_drop * 100, 2))
            })
            
        # Sort explanations by importance
        explanations = sorted(explanations, key=lambda x: x['importance'], reverse=True)
        
        # Assign category flags based on contribution
        for rank, exp in enumerate(explanations):
            if rank == 0:
                exp['impact'] = 'High Impact'
                exp['description'] = f"Primary diagnostic indicator. Setting this symptom to absent drops model confidence by {exp['percentage_drop']}%."
            elif rank == 1:
                exp['impact'] = 'Medium Impact'
                exp['description'] = f"Secondary supportive feature. Contributes {exp['percentage_drop']}% to the final prediction."
            else:
                exp['impact'] = 'Supporting'
                exp['description'] = f"General symptom. Adds marginal weight ({exp['percentage_drop']}%) during neural weights summation."
                
        return {
            'prognosis': prognosis,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'explanations': explanations
        }
