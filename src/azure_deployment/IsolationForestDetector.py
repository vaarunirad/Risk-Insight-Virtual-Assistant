import pandas as pd
import numpy as np
import logging
import time
from sklearn.ensemble import IsolationForest
import joblib
import os
import io

class IsolationForestDetector:
    """Anomaly detection using Isolation Forest algorithm.
    
    This detector uses machine learning to identify anomalies based on feature isolation.
    It's particularly effective for multivariate anomaly detection and can find anomalies
    that statistical methods might miss.
    """
    
    def __init__(self, config=None, logger=None):
        """Initialize the isolation forest detector"""
        self.default_config = {
            'temperature': {
                'contamination': 0.03,    # Expected proportion of anomalies
                'n_estimators': 100,      # Number of trees
                'max_samples': 'auto',    # Samples to draw for each tree
                'random_state': 42,       # For reproducibility
                'batch_size': 50000,      # Process in batches
                'features': ['temperature_celsius', 'z_score'],  # Features to use
                'min_data_points': 20     # Minimum data points required
            },
            'battery': {
                'contamination': 0.02,    # Lower contamination for battery
                'n_estimators': 100,
                'max_samples': 'auto',
                'random_state': 42,
                'batch_size': 30000,
                'features': ['battery_voltage', 'z_score'],
                'min_data_points': 15
            },
            'vibration': {
                'contamination': 0.03,
                'n_estimators': 100,
                'max_samples': 'auto',
                'random_state': 42,
                'batch_size': 10000,
                'features': ['vibration_magnitude', 'z_score'],
                'min_data_points': 15
            },
            'motion': {
                'contamination': 0.05,    # Higher contamination for motion
                'n_estimators': 100,
                'max_samples': 'auto',
                'random_state': 42,
                'batch_size': 5000,
                'features': ['max_magnitude', 'daily_event_count'],
                'min_data_points': 10
            }
        }
        
        self.config = config or self.default_config
        self.logger = logger or logging.getLogger('isolation_forest_detector')
        self.results = {}
        self.models = {}
        self.models_dir = '.models/isolation_forest'
        os.makedirs(self.models_dir, exist_ok=True)
        
    def detect(self, df, sensor_type):
        """Detect anomalies using Isolation Forest algorithm"""
        if df is None or df.empty:
            self.logger.warning(f"Empty dataframe for {sensor_type}, skipping isolation forest detection")
            return df
            
        self.logger.info(f"Starting isolation forest detection for {sensor_type} data")
        
        try:
            # Validate required columns
            required_columns = ['sensor_id']
            if sensor_type == 'temperature':
                required_columns.append('temperature_celsius')
            elif sensor_type == 'battery':
                required_columns.append('battery_voltage')
            elif sensor_type == 'vibration':
                required_columns.append('vibration_magnitude')
            
            # Check if all required columns exist
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns for {sensor_type} detection: {missing_columns}")
                return df
            
            # Create a copy to avoid warnings
            df = df.copy()
            
            # Add isolation forest anomaly column
            df['isolation_forest_anomaly'] = False
            df['isolation_forest_score'] = 0.0
            
            # Get configuration for this sensor type
            config = self.config.get(sensor_type, self.default_config['temperature'])
            batch_size = config.get('batch_size', 50000)
            min_data_points = config.get('min_data_points', 10)
            
            # Process each sensor individually
            anomaly_count = 0
            total_processed = 0
            
            for sensor_id in df['sensor_id'].unique():
                try:
                    # Get sensor data
                    sensor_df = df[df['sensor_id'] == sensor_id].copy()
                    
                    # Skip if not enough data
                    if len(sensor_df) < min_data_points:
                        self.logger.debug(f"Skipping sensor {sensor_id} - insufficient data points ({len(sensor_df)})")
                        continue
                    
                    # Get features for this sensor
                    features = self.get_features(sensor_df, config.get('features', []), sensor_type)
                    
                    # Skip if not enough valid features
                    if not features or len(features) == 0:
                        self.logger.debug(f"Skipping sensor {sensor_id} - no valid features found")
                        continue
                    
                    # Prepare feature matrix
                    # Fill missing values with the mean to avoid NaN errors
                    for feature in features:
                        if sensor_df[feature].isna().any():
                            mean_value = sensor_df[feature].mean()
                            sensor_df[feature] = sensor_df[feature].fillna(mean_value)
                    
                    X = sensor_df[features].values
                    
                    # Skip if no valid data or constant features
                    if X.shape[0] == 0 or X.shape[1] == 0:
                        self.logger.debug(f"Skipping sensor {sensor_id} - empty feature matrix")
                        continue
                    
                    # Check for constant features
                    std_values = np.std(X, axis=0)
                    if np.any(std_values == 0):
                        self.logger.debug(f"Skipping sensor {sensor_id} - has constant features")
                        continue
                    
                    try:
                        # Create and fit the model
                        contamination = min(0.5, max(0.001, config.get('contamination', 0.03)))
                        
                        model = IsolationForest(
                            contamination=contamination,
                            n_estimators=config.get('n_estimators', 100),
                            max_samples=config.get('max_samples', 'auto'),
                            random_state=config.get('random_state', 42),
                            n_jobs=-1  # Use all available cores
                        )
                        
                        # Fit the model
                        model.fit(X)
                        
                        # Save the model
                        model_key = f"{sensor_type}_{sensor_id}"
                        self.models[model_key] = model
                        
                        # Predict anomalies
                        y_pred = model.predict(X)
                        scores = model.decision_function(X)
                        
                        # Convert scores to anomaly scores (0-1 range, higher means more anomalous)
                        min_score = scores.min() if len(scores) > 0 else -1
                        max_score = scores.max() if len(scores) > 0 else 1
                        
                        # Avoid division by zero
                        if max_score - min_score > 1e-10:
                            anomaly_scores = 1 - (scores - min_score) / (max_score - min_score)
                        else:
                            anomaly_scores = np.zeros(len(scores))
                        
                        anomalies = (y_pred == -1)
                        
                        # Update dataframe
                        df.loc[sensor_df.index[anomalies], 'isolation_forest_anomaly'] = True
                        df.loc[sensor_df.index, 'isolation_forest_score'] = anomaly_scores
                        
                        # Count anomalies
                        anomaly_count += anomalies.sum()
                        total_processed += len(sensor_df)
                        
                    except Exception as e:
                        self.logger.warning(f"Error in isolation forest model for {sensor_id}: {str(e)}")
                except Exception as e:
                    self.logger.warning(f"Error processing sensor {sensor_id} for isolation forest: {str(e)}")
            
            # Calculate anomaly percentage
            anomaly_percentage = anomaly_count / len(df) * 100 if len(df) > 0 else 0
            
            # Log results
            self.logger.info(f"Isolation forest detection results for {sensor_type}: {anomaly_count} anomalies, {anomaly_percentage:.2f}%")
            
            # Store results
            self.results[sensor_type] = {
                'anomaly_count': int(anomaly_count),
                'total_records': len(df),
                'anomaly_percentage': float(anomaly_percentage),
                'sensors_processed': len(self.models)
            }
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in isolation forest detection for {sensor_type}: {str(e)}")
            # Return the original dataframe to allow the pipeline to continue
            return df

    def get_features(self, df, feature_list, sensor_type):
        """Get valid features for the isolation forest model"""
        valid_features = []
        
        # Check both renamed and original column names
        feature_mapping = {
            'temperature_celsius': 'TemperatureInCelsius',
            'battery_voltage': 'BatteryVoltage',
            'vibration_magnitude': 'MagnitudeRMSInMilliG',
            'z_score': 'z_score'
        }
        
        # Try to use columns from feature list first
        for feature in feature_list:
            # Check if feature exists directly
            if feature in df.columns:
                # Verify it has valid values
                if not df[feature].isna().all() and df[feature].nunique() > 1:
                    valid_features.append(feature)
            # Check if original column name exists
            elif feature in feature_mapping and feature_mapping[feature] in df.columns:
                orig_col = feature_mapping[feature]
                if not df[orig_col].isna().all() and df[orig_col].nunique() > 1:
                    valid_features.append(orig_col)
        
        # If no features found, try to find appropriate columns based on sensor type
        if not valid_features:
            if sensor_type == 'temperature':
                temp_cols = [col for col in df.columns if 'temp' in col.lower() or 'celsius' in col.lower()]
                for col in temp_cols:
                    if not df[col].isna().all() and df[col].nunique() > 1:
                        valid_features.append(col)
            elif sensor_type == 'battery':
                batt_cols = [col for col in df.columns if 'batt' in col.lower() or 'volt' in col.lower()]
                for col in batt_cols:
                    if not df[col].isna().all() and df[col].nunique() > 1:
                        valid_features.append(col)
            elif sensor_type == 'vibration':
                vib_cols = [col for col in df.columns if 'vibration' in col.lower() or 'magnitude' in col.lower()]
                for col in vib_cols:
                    if not df[col].isna().all() and df[col].nunique() > 1:
                        valid_features.append(col)
            elif sensor_type == 'motion':
                motion_cols = [col for col in df.columns if 'motion' in col.lower() or 'event' in col.lower() or 'count' in col.lower()]
                for col in motion_cols:
                    if not df[col].isna().all() and df[col].nunique() > 1:
                        valid_features.append(col)
        
        # Add time-based features if event_time is available
        if 'event_time' in df.columns:
            try:
                # Ensure event_time is datetime
                if not pd.api.types.is_datetime64_any_dtype(df['event_time']):
                    df['event_time'] = pd.to_datetime(df['event_time'])
                
                # Extract hour of day and day of week as separate numeric columns
                # This prevents operations on Timestamp objects
                df['hour_of_day'] = df['event_time'].dt.hour
                if 'hour_of_day' in df.columns and df['hour_of_day'].nunique() > 1:
                    valid_features.append('hour_of_day')
                
                df['day_of_week'] = df['event_time'].dt.dayofweek
                if 'day_of_week' in df.columns and df['day_of_week'].nunique() > 1:
                    valid_features.append('day_of_week')
            except Exception as e:
                self.logger.debug(f"Error adding time features: {str(e)}")
                
        # Make sure all features exist in the dataframe
        valid_features = [f for f in valid_features if f in df.columns]
                
        # Log the features being used
        if valid_features:
            self.logger.debug(f"Using features: {valid_features}")
        else:
            self.logger.warning(f"No valid features found for isolation forest detection")
            
        return valid_features

    def save_models(self):
        """Save trained models to disk"""
        if not self.models:
            self.logger.info("No models to save")
            return
            
        try:
            os.makedirs(self.models_dir, exist_ok=True)
            
            saved_count = 0
            for model_key, model in self.models.items():
                try:
                    model_path = os.path.join(self.models_dir, f"{model_key}.joblib")
                    joblib.dump(model, model_path)
                    saved_count += 1
                except Exception as e:
                    self.logger.warning(f"Error saving model {model_key}: {str(e)}")
                
            self.logger.info(f"Saved {saved_count} isolation forest models")
        except Exception as e:
            self.logger.error(f"Error saving isolation forest models: {str(e)}")
    
    def save_models_to_blob(self):
        """Save trained models to blob storage"""
        if not self.models:
            self.logger.info("No models to save to blob storage")
            return
        
        try:
            from azure.storage.blob import BlobServiceClient
            
            # Get storage connection details
            connection_string = os.environ.get('STORAGE_CONNECTION_STRING')
            container_name = os.environ.get('PROCESSED_CONTAINER', 'processed-data')
            
            if not connection_string:
                self.logger.error("Missing storage connection string")
                return
            
            # Initialize blob service
            blob_service = BlobServiceClient.from_connection_string(connection_string)
            container_client = blob_service.get_container_client(container_name)
            
            # Save each model to blob storage
            saved_count = 0
            timestamp = time.strftime('%Y%m%d%H%M%S')
            
            for model_key, model in self.models.items():
                try:
                    # Save model to memory buffer
                    model_buffer = io.BytesIO()
                    joblib.dump(model, model_buffer)
                    model_buffer.seek(0)
                    
                    # Upload to blob storage
                    blob_name = f"models/isolation_forest/{timestamp}/{model_key}.joblib"
                    blob_client = container_client.get_blob_client(blob_name)
                    blob_client.upload_blob(model_buffer, overwrite=True)
                    
                    saved_count += 1
                except Exception as e:
                    self.logger.warning(f"Error saving model {model_key} to blob: {str(e)}")
            
            self.logger.info(f"Saved {saved_count} isolation forest models to blob storage")
        except Exception as e:
            self.logger.error(f"Error saving models to blob storage: {str(e)}")

    def load_models(self):
        """Load trained models from disk"""
        try:
            if not os.path.exists(self.models_dir):
                self.logger.info("No models directory found")
                return
                
            model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.joblib')]
            
            loaded_count = 0
            for model_file in model_files:
                try:
                    model_key = model_file.replace('.joblib', '')
                    model_path = os.path.join(self.models_dir, model_file)
                    
                    self.models[model_key] = joblib.load(model_path)
                    loaded_count += 1
                except Exception as e:
                    self.logger.warning(f"Error loading model {model_file}: {str(e)}")
                
            self.logger.info(f"Loaded {loaded_count} isolation forest models")
        except Exception as e:
            self.logger.error(f"Error loading isolation forest models: {str(e)}")

    def get_results(self, sensor_type=None):
        """Get detection results"""
        if sensor_type:
            return self.results.get(sensor_type, {})
        return self.results