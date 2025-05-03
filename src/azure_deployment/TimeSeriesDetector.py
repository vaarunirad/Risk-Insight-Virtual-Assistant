import pandas as pd
import numpy as np
import logging
import time
import gc  # For garbage collection
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import os
import io
import warnings
warnings.filterwarnings("ignore")

class TimeSeriesDetector:
    """Anomaly detection using time series forecasting models.
    
    This detector uses statistical time series models to identify temporal
    pattern anomalies that might be missed by other detection methods.
    It supports:
    1. Seasonal ARIMA models for seasonal data
    2. Exponential smoothing for trend detection
    3. Prediction interval-based anomaly detection
    """
    
    def __init__(self, config=None, logger=None):
        """Initialize the time series detector"""
        self.default_config = {
            'temperature': {
                'min_training_points': 48,  # Minimum points needed (4 hours for 5-min data)
                'prediction_interval': 0.95, # 95% prediction interval
                'error_threshold': 2.5,     # Z-score threshold for prediction errors
                'batch_size': 200,         # Reduced batch size for memory efficiency
                'max_sensors': 10,         # Maximum sensors to process at once
                'use_seasonal': True,       # Use seasonal models for temperature
                'seasonal_period': 288,     # Daily seasonality (5-min data: 12 * 24 = 288)
                'diff_order': 1,            # Differencing order for SARIMA
                'model_params': {
                    'auto_arima': False,    # Auto ARIMA too slow for large datasets
                    'fallback_to_ets': True # Use exponential smoothing
                },
                'reuse_models': True        # Reuse previously trained models
            },
            'battery': {
                'min_training_points': 24,  # Minimum points needed (6 hours for 15-min data)
                'prediction_interval': 0.90, # 90% prediction interval
                'error_threshold': 3.0,     # Higher threshold for battery data
                'batch_size': 200,
                'max_sensors': 15,         # Maximum sensors to process at once
                'use_seasonal': False,      # Battery doesn't typically have strong seasonality
                'model_params': {
                    'auto_arima': False,    # Skip autoarima for battery
                    'use_holt_winters': True # Use Holt-Winters for battery (captures trend)
                },
                'reuse_models': True
            },
            'vibration': {
                'min_training_points': 24,  # Minimum points needed (1 day for hourly data)
                'prediction_interval': 0.98, # 98% prediction interval (wider for vibration)
                'error_threshold': 3.5,     # Higher threshold for vibration data
                'batch_size': 200,
                'max_sensors': 20,         # Maximum sensors to process at once
                'use_seasonal': True,       # Vibration can have daily patterns
                'seasonal_period': 24,      # Daily seasonality for hourly data
                'model_params': {
                    'auto_arima': False,
                    'use_holt_winters': True
                },
                'reuse_models': True
            },
            'motion': {
                'min_training_points': 14,  # Minimum points needed (2 weeks for daily data)
                'prediction_interval': 0.90,
                'error_threshold': 2.5,
                'batch_size': 200,
                'max_sensors': 50,         # Maximum sensors to process at once
                'use_seasonal': True,
                'seasonal_period': 7,       # Weekly seasonality for daily data
                'model_params': {
                    'use_holt_winters': True
                },
                'reuse_models': True
            }
        }
        
        self.config = config or self.default_config
        self.logger = logger or logging.getLogger('time_series_detector')
        self.results = {}
        self.models = {}
        self.models_dir = '.models/time_series'
        os.makedirs(self.models_dir, exist_ok=True)
        
    def detect(self, df, sensor_type):
        """Detect anomalies using time series forecasting models with memory optimization"""
        if df is None or df.empty:
            self.logger.warning(f"Empty dataframe for {sensor_type}, skipping time series detection")
            return df
            
        self.logger.info(f"Starting time series anomaly detection for {sensor_type} data")
        
        try:
            # Validate required columns
            required_columns = ['sensor_id', 'event_time']
            if sensor_type == 'temperature':
                required_columns.append('temperature_celsius')
            elif sensor_type == 'battery':
                required_columns.append('battery_voltage')
            elif sensor_type == 'vibration':
                required_columns.append('vibration_magnitude')
            
            # Check for missing columns and handle gracefully
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns for {sensor_type} time series detection: {missing_columns}")
                
                # Try to find alternative columns
                alternate_found = False
                if 'temperature_celsius' in missing_columns and 'TemperatureInCelsius' in df.columns:
                    df['temperature_celsius'] = df['TemperatureInCelsius']
                    missing_columns.remove('temperature_celsius')
                    alternate_found = True
                    
                if 'battery_voltage' in missing_columns and 'BatteryVoltage' in df.columns:
                    df['battery_voltage'] = df['BatteryVoltage']
                    missing_columns.remove('battery_voltage')
                    alternate_found = True
                    
                if 'vibration_magnitude' in missing_columns and 'MagnitudeRMSInMilliG' in df.columns:
                    df['vibration_magnitude'] = df['MagnitudeRMSInMilliG']
                    missing_columns.remove('vibration_magnitude')
                    alternate_found = True
                    
                if 'event_time' in missing_columns and 'EventTimeUtc' in df.columns:
                    df['event_time'] = pd.to_datetime(df['EventTimeUtc'])
                    missing_columns.remove('event_time')
                    alternate_found = True
                    
                if 'sensor_id' in missing_columns and 'SensorId' in df.columns:
                    df['sensor_id'] = df['SensorId']
                    missing_columns.remove('sensor_id')
                    alternate_found = True
                
                # If we still have missing columns after trying alternates, return original
                if missing_columns and not alternate_found:
                    return df
            
            # Create a copy to avoid warnings
            df = df.copy()
            
            # Add time series anomaly columns
            df['time_series_anomaly'] = False
            df['time_series_score'] = 0.0
            df['forecast_value'] = np.nan
            df['forecast_lower'] = np.nan
            df['forecast_upper'] = np.nan
            
            # Get configuration for this sensor type
            config = self.config.get(sensor_type, self.default_config['temperature'])
            min_training_points = config.get('min_training_points', 24)
            max_sensors = config.get('max_sensors', 10)  # Limit number of sensors processed
            
            # Get value column for this sensor type
            value_col = self.get_value_column(df, sensor_type)
            if not value_col:
                self.logger.error(f"Could not determine value column for {sensor_type}")
                return df
                
            # Process each sensor individually, but limit the number of sensors
            anomaly_count = 0
            processed_sensors = 0
            
            # Get unique sensor IDs
            unique_sensors = df['sensor_id'].unique()
            
            # Limit the number of sensors to process for memory efficiency
            if len(unique_sensors) > max_sensors:
                self.logger.warning(f"Limiting time series detection to {max_sensors} sensors for memory efficiency")
                # Only process the sensors with the most data points
                sensor_counts = df.groupby('sensor_id').size()
                top_sensors = sensor_counts.nlargest(max_sensors).index.tolist()
                unique_sensors = top_sensors
            
            for sensor_id in unique_sensors:
                try:
                    # Get sensor data
                    sensor_df = df[df['sensor_id'] == sensor_id].copy()
                    
                    # Skip if not enough data
                    if len(sensor_df) < min_training_points:
                        self.logger.debug(f"Skipping sensor {sensor_id} - insufficient data points ({len(sensor_df)})")
                        continue
                    
                    # Skip if value column doesn't exist or is all NA
                    if value_col not in sensor_df.columns or sensor_df[value_col].isna().all():
                        self.logger.debug(f"Skipping sensor {sensor_id} - missing value column {value_col}")
                        continue
                        
                    # Ensure data is sorted by time
                    if 'event_time' in sensor_df.columns:
                        try:
                            # Convert to datetime if not already
                            if not pd.api.types.is_datetime64_any_dtype(sensor_df['event_time']):
                                sensor_df['event_time'] = pd.to_datetime(sensor_df['event_time'])
                            
                            # Sort by time
                            sensor_df = sensor_df.sort_values('event_time')
                        except Exception as e:
                            self.logger.warning(f"Error sorting data for sensor {sensor_id}: {str(e)}")
                            # Continue without sorting if it fails
                            
                    # Process in smaller batches for large sensors
                    batch_size = config.get('batch_size', 200)  # Smaller batch size for memory efficiency
                    
                    # If sensor data is large, process in batches
                    if len(sensor_df) > batch_size * 2:  # Only batch if it's substantially larger than batch size
                        self.logger.debug(f"Processing sensor {sensor_id} in batches")
                        
                        # Extract time series data for training
                        ts_data = sensor_df[value_col].values[:min(len(sensor_df), 2000)]  # Limit training data size
                        
                        # Get model key
                        model_key = f"{sensor_type}_{sensor_id}"
                        
                        # Train or load model once
                        try:
                            # Try to use existing model if available
                            if config.get('reuse_models', True) and model_key in self.models:
                                model_info = self.models[model_key]
                            else:
                                # Train a new model
                                model_info = self.train_model(ts_data, sensor_type, config)
                                if model_info:
                                    self.models[model_key] = model_info
                                else:
                                    self.logger.debug(f"Skipping sensor {sensor_id} - could not train model")
                                    continue
                        except Exception as e:
                            self.logger.warning(f"Error training model for sensor {sensor_id}: {str(e)}")
                            continue
                        
                        # Process data in batches to save memory
                        batch_anomalies_count = 0
                        for i in range(0, len(sensor_df), batch_size):
                            # Get batch
                            batch = sensor_df.iloc[i:i+batch_size].copy()
                            
                            batch_ts_data = batch[value_col].values
                            
                            # Skip if data is constant
                            if np.std(batch_ts_data) == 0:
                                continue
                            
                            # Make predictions on this batch
                            try:
                                predictions = self.predict_with_model(model_info, batch_ts_data)
                                if predictions is None:
                                    continue
                                    
                                forecast, lower, upper, errors, scores = predictions
                                
                                # Store forecasts in dataframe for this batch
                                df.loc[batch.index, 'forecast_value'] = forecast
                                df.loc[batch.index, 'forecast_lower'] = lower
                                df.loc[batch.index, 'forecast_upper'] = upper
                                
                                # Mark anomalies
                                anomalies = (scores > 0.5)
                                df.loc[batch.index[anomalies], 'time_series_anomaly'] = True
                                df.loc[batch.index, 'time_series_score'] = scores
                                
                                # Count anomalies
                                batch_anomalies_count += anomalies.sum()
                            
                                # Force garbage collection to free memory
                                del batch, forecast, lower, upper, errors, scores, batch_ts_data
                                gc.collect()
                            
                            except Exception as e:
                                self.logger.warning(f"Error in batch processing for sensor {sensor_id}: {str(e)}")
                                continue
                                
                        # Update total anomaly count
                        anomaly_count += batch_anomalies_count
                    
                    else:
                        # Process small sensors all at once
                        # Extract time series data
                        ts_data = sensor_df[value_col].values
                        
                        # Skip if data is constant
                        if np.std(ts_data) == 0:
                            self.logger.debug(f"Skipping sensor {sensor_id} - constant data")
                            continue
                            
                        # Get model key
                        model_key = f"{sensor_type}_{sensor_id}"
                        
                        try:
                            # Try to use existing model if available
                            if config.get('reuse_models', True) and model_key in self.models:
                                model_info = self.models[model_key]
                                predictions = self.predict_with_model(model_info, ts_data)
                            else:
                                # Train a new model
                                model_info = self.train_model(ts_data, sensor_type, config)
                                if model_info:
                                    self.models[model_key] = model_info
                                    predictions = self.predict_with_model(model_info, ts_data)
                                else:
                                    self.logger.debug(f"Skipping sensor {sensor_id} - could not train model")
                                    continue
                                    
                            # Skip if predictions couldn't be generated
                            if predictions is None:
                                self.logger.debug(f"Skipping sensor {sensor_id} - predictions failed")
                                continue
                                
                            forecast, lower, upper, errors, scores = predictions
                            
                            # Make sure all arrays are same length as original data
                            if len(forecast) != len(sensor_df):
                                # Truncate or pad to match the original data length
                                if len(forecast) > len(sensor_df):
                                    forecast = forecast[:len(sensor_df)]
                                    lower = lower[:len(sensor_df)]
                                    upper = upper[:len(sensor_df)]
                                    errors = errors[:len(sensor_df)]
                                    scores = scores[:len(sensor_df)]
                                else:
                                    # Pad with last value
                                    pad_length = len(sensor_df) - len(forecast)
                                    last_forecast = forecast[-1] if len(forecast) > 0 else np.mean(ts_data)
                                    last_lower = lower[-1] if len(lower) > 0 else (last_forecast - np.std(ts_data))
                                    last_upper = upper[-1] if len(upper) > 0 else (last_forecast + np.std(ts_data))
                                    
                                    forecast = np.append(forecast, np.full(pad_length, last_forecast))
                                    lower = np.append(lower, np.full(pad_length, last_lower))
                                    upper = np.append(upper, np.full(pad_length, last_upper))
                                    errors = np.append(errors, np.full(pad_length, 0))
                                    scores = np.append(scores, np.full(pad_length, 0))
                            
                            # Store forecasts in dataframe
                            df.loc[sensor_df.index, 'forecast_value'] = forecast
                            df.loc[sensor_df.index, 'forecast_lower'] = lower
                            df.loc[sensor_df.index, 'forecast_upper'] = upper
                            
                            # Mark anomalies
                            anomalies = (scores > 0.5)
                            df.loc[sensor_df.index[anomalies], 'time_series_anomaly'] = True
                            df.loc[sensor_df.index, 'time_series_score'] = scores
                            
                            # Count anomalies
                            anomaly_count += anomalies.sum()
                            
                        except Exception as e:
                            self.logger.warning(f"Error in time series detection for sensor {sensor_id}: {str(e)}")
                    
                    processed_sensors += 1
                    
                    # Force garbage collection after each sensor
                    gc.collect()
                    
                except Exception as e:
                    self.logger.warning(f"Error processing sensor {sensor_id} for time series detection: {str(e)}")
            
            # Calculate anomaly percentage
            anomaly_percentage = anomaly_count / len(df) * 100 if len(df) > 0 else 0
            
            # Log results
            self.logger.info(f"Time series detection results for {sensor_type}: {anomaly_count} anomalies, {anomaly_percentage:.2f}%")
            
            self.results[sensor_type] = {
                'anomaly_count': int(anomaly_count),
                'total_records': len(df),
                'anomaly_percentage': float(anomaly_percentage),
                'sensors_processed': processed_sensors
            }
            
            # Save models to blob storage
            try:
                self.save_models_to_blob()
            except Exception as e:
                self.logger.error(f"Error saving time series models to blob storage: {str(e)}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in time series detection for {sensor_type}: {str(e)}")
            # Return the original dataframe to allow the pipeline to continue
            return df
        
    def train_model(self, data, sensor_type, config):
        """Train a time series model for a sensor"""
        try:
            # Get configuration
            use_seasonal = config.get('use_seasonal', False)
            seasonal_period = config.get('seasonal_period', 24)
            model_params = config.get('model_params', {})
            
            # Limit training data size
            max_training_points = min(500, len(data))  # Cap training data size for memory efficiency
            data = data[:max_training_points]
            
            # Split data into train/test for better model evaluation
            train_size = int(len(data) * 0.8)
            train_data = data[:train_size]
            
            # Check if we have enough training data
            if len(train_data) < config.get('min_training_points', 24):
                self.logger.debug("Insufficient training data")
                return None
            
            # 1. First try simple exponential smoothing (least memory intensive)
            if model_params.get('fallback_to_ets', True):
                try:
                    # Create model
                    model = ExponentialSmoothing(
                        train_data,
                        trend=None,
                        seasonal=None
                    )
                    
                    # Fit model
                    fit_result = model.fit()
                    
                    # Return model info
                    return {
                        'type': 'simple_ets',
                        'model': fit_result,
                        'seasonal': False
                    }
                except:
                    pass
            
            # 2. Try Holt-Winters Exponential Smoothing if enabled
            if model_params.get('use_holt_winters', True):
                try:
                    # Determine seasonality parameters
                    seasonal = 'add' if use_seasonal else None
                    seasonal_periods = seasonal_period if use_seasonal else None
                    
                    # Create model
                    model = ExponentialSmoothing(
                        train_data,
                        trend='add',
                        seasonal=seasonal,
                        seasonal_periods=seasonal_periods,
                        damped=True
                    )
                    
                    # Fit model with limited iterations to save memory
                    fit_result = model.fit(optimized=False, remove_bias=True)
                    
                    # Return model info
                    return {
                        'type': 'ets',
                        'model': fit_result,
                        'seasonal': use_seasonal,
                        'seasonal_period': seasonal_period
                    }
                except:
                    pass
            
            # 3. Fallback to moving average as a last resort
            try:
                # Calculate mean value for constant model
                mean_value = np.mean(train_data)
                
                # Return a simple model that just uses the mean
                return {
                    'type': 'constant',
                    'model': None,  # No actual model object
                    'mean': mean_value,
                    'std': np.std(train_data)
                }
            except:
                return None
                
        except Exception as e:
            self.logger.error(f"Error training time series model: {str(e)}")
            return None
            
    def predict_with_model(self, model_info, data):
        """Make predictions with a trained model"""
        try:
            model_type = model_info.get('type', 'unknown')
            
            # Simple constant model is most memory efficient
            if model_type == 'constant':
                # Simple constant model just returns the mean value
                mean_value = model_info.get('mean', np.mean(data))
                std_value = model_info.get('std', np.std(data))
                
                # Use z value for 95% confidence
                z_value = 1.96
                
                # Create constant arrays
                fitted = np.full(len(data), mean_value)
                lower = np.full(len(data), mean_value - z_value * std_value)
                upper = np.full(len(data), mean_value + z_value * std_value)
                
                # Calculate errors and scores
                errors = data - fitted
                abs_errors = np.abs(errors)
                max_error = np.max(abs_errors) if len(abs_errors) > 0 else 1
                scores = abs_errors / max_error if max_error > 0 else abs_errors
                
                return fitted, lower, upper, errors, scores
                
            # Exponential smoothing models
            elif model_type == 'ets' or model_type == 'simple_ets':
                model = model_info.get('model')
                if model is None:
                    self.logger.warning("Missing model object for ETS")
                    return None
                
                # Get fitted values
                fitted = model.fittedvalues
                
                # Calculate residuals for prediction intervals
                resid = model.resid
                resid_std = np.std(resid) if len(resid) > 0 else 0.1
                
                # Use z value for 95% confidence
                z_value = 1.96
                
                # Calculate intervals
                lower = fitted - z_value * resid_std
                upper = fitted + z_value * resid_std
                
                # Convert to numpy arrays if needed
                if hasattr(fitted, 'values'):
                    fitted = fitted.values
                if hasattr(lower, 'values'):
                    lower = lower.values  
                if hasattr(upper, 'values'):
                    upper = upper.values
                
                # Pad with the last value for any remaining data points
                if len(fitted) < len(data):
                    # Get the last fitted value
                    last_fitted = fitted[-1] if len(fitted) > 0 else np.mean(data)
                    
                    # Create padding of appropriate length
                    pad_length = len(data) - len(fitted)
                    
                    # Add padding
                    fitted = np.append(fitted, np.full(pad_length, last_fitted))
                    lower = np.append(lower, np.full(pad_length, lower[-1] if len(lower) > 0 else (last_fitted - resid_std)))
                    upper = np.append(upper, np.full(pad_length, upper[-1] if len(upper) > 0 else (last_fitted + resid_std)))
                
                # Take just what we need if arrays are too long
                if len(fitted) > len(data):
                    fitted = fitted[:len(data)]
                    lower = lower[:len(data)]
                    upper = upper[:len(data)]
                
                # Calculate errors
                errors = data - fitted
                
                # Calculate normalized scores (0-1 range, higher means more anomalous)
                abs_errors = np.abs(errors)
                max_error = np.max(abs_errors) if len(abs_errors) > 0 else 1
                scores = abs_errors / max_error if max_error > 0 else abs_errors
                
                # Adjust scores based on prediction intervals
                for i in range(len(data)):
                    if i < len(lower) and i < len(upper):
                        if not np.isnan(lower[i]) and not np.isnan(upper[i]):
                            if data[i] < lower[i] or data[i] > upper[i]:
                                scores[i] = max(scores[i], 0.8)  # Simpler calculation to save memory
                
                return fitted, lower, upper, errors, scores
                
            else:
                self.logger.warning(f"Unsupported model type for memory optimization: {model_type}")
                
                # Fall back to constant model
                mean_value = np.mean(data)
                std_value = np.std(data)
                z_value = 1.96
                
                fitted = np.full(len(data), mean_value)
                lower = np.full(len(data), mean_value - z_value * std_value)
                upper = np.full(len(data), mean_value + z_value * std_value)
                
                errors = data - fitted
                abs_errors = np.abs(errors)
                max_error = np.max(abs_errors) if len(abs_errors) > 0 else 1
                scores = abs_errors / max_error if max_error > 0 else abs_errors
                
                return fitted, lower, upper, errors, scores
                
        except Exception as e:
            self.logger.error(f"Error making predictions with time series model: {str(e)}")
            return None
            
    def get_value_column(self, df, sensor_type):
        """Get the value column for a sensor type"""
        # Try the renamed column first
        if sensor_type == 'temperature':
            if 'temperature_celsius' in df.columns:
                return 'temperature_celsius'
            elif 'TemperatureInCelsius' in df.columns:
                return 'TemperatureInCelsius'
        elif sensor_type == 'battery':
            if 'battery_voltage' in df.columns:
                return 'battery_voltage'
            elif 'BatteryVoltage' in df.columns:
                return 'BatteryVoltage'
        elif sensor_type == 'vibration':
            if 'vibration_magnitude' in df.columns:
                return 'vibration_magnitude'
            elif 'MagnitudeRMSInMilliG' in df.columns:
                return 'MagnitudeRMSInMilliG'
        elif sensor_type == 'motion':
            if 'max_magnitude' in df.columns:
                return 'max_magnitude'
            elif 'daily_event_count' in df.columns:
                return 'daily_event_count'
            elif 'MaxMagnitudeRMSInMilliG' in df.columns:
                return 'MaxMagnitudeRMSInMilliG'
        
        # If we don't find expected columns, look for any column that might work
        if sensor_type == 'temperature':
            temperature_cols = [col for col in df.columns if 'temp' in col.lower() or 'celsius' in col.lower()]
            if temperature_cols:
                return temperature_cols[0]
        elif sensor_type == 'battery':
            battery_cols = [col for col in df.columns if 'batt' in col.lower() or 'volt' in col.lower()]
            if battery_cols:
                return battery_cols[0]
        elif sensor_type == 'vibration':
            vibration_cols = [col for col in df.columns if 'vibration' in col.lower() or 'magnitude' in col.lower() or 'rms' in col.lower()]
            if vibration_cols:
                return vibration_cols[0]
        elif sensor_type == 'motion':
            motion_cols = [col for col in df.columns if 'motion' in col.lower() or 'event' in col.lower() or 'count' in col.lower() or 'magnitude' in col.lower()]
            if motion_cols:
                return motion_cols[0]
            
        # If no column found, log warning and return None
        self.logger.warning(f"Could not find a suitable value column for {sensor_type}")
        return None
        
    def save_models_to_blob(self):
        """Save trained models to blob storage"""
        if not self.models:
            self.logger.info("No time series models to save to blob storage")
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
            
            for model_key, model_info in self.models.items():
                try:
                    # Skip "constant" models as they are just statistics
                    if model_info.get('type') == 'constant':
                        continue
                    
                    # Save model to memory buffer
                    model_buffer = io.BytesIO()
                    joblib.dump(model_info, model_buffer)
                    model_buffer.seek(0)
                    
                    # Upload to blob storage
                    blob_name = f"models/time_series/{timestamp}/{model_key}.pkl"
                    blob_client = container_client.get_blob_client(blob_name)
                    blob_client.upload_blob(model_buffer, overwrite=True)
                    
                    saved_count += 1
                except Exception as e:
                    self.logger.warning(f"Error saving model {model_key} to blob: {str(e)}")
            
            self.logger.info(f"Saved {saved_count} time series models to blob storage")
        except Exception as e:
            self.logger.error(f"Error saving time series models to blob storage: {str(e)}")
        
    def save_models(self, output_dir=None):
        """Save trained models to disk"""
        if output_dir is None:
            output_dir = self.models_dir
            
        if not self.models:
            self.logger.info("No models to save")
            return
            
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            saved_count = 0
            for model_key, model_info in self.models.items():
                try:
                    # Skip "constant" models as they are just statistics
                    if model_info.get('type') == 'constant':
                        continue
                        
                    # Export model to disk
                    output_path = os.path.join(output_dir, f"timeseries_{model_key}.pkl")
                    joblib.dump(model_info, output_path)
                    saved_count += 1
                except Exception as e:
                    self.logger.warning(f"Error saving model {model_key}: {str(e)}")
                
            self.logger.info(f"Saved {saved_count} time series models to {output_dir}")
        except Exception as e:
            self.logger.error(f"Error saving time series models: {str(e)}")
        
    def load_models(self, input_dir=None):
        """Load trained models from disk"""
        if input_dir is None:
            input_dir = self.models_dir
            
        if not os.path.exists(input_dir):
            self.logger.info("No models directory found")
            return
            
        try:
            model_files = [f for f in os.listdir(input_dir) if f.startswith('timeseries_') and f.endswith('.pkl')]
            
            if not model_files:
                self.logger.info("No time series model files found")
                return
                
            loaded_count = 0
            for model_file in model_files:
                try:
                    model_key = model_file.replace('timeseries_', '').replace('.pkl', '')
                    model_path = os.path.join(input_dir, model_file)
                    
                    self.models[model_key] = joblib.load(model_path)
                    loaded_count += 1
                except Exception as e:
                    self.logger.warning(f"Error loading model {model_file}: {str(e)}")
                
            self.logger.info(f"Loaded {loaded_count} time series models")
        except Exception as e:
            self.logger.error(f"Error loading time series models: {str(e)}")
        
    def get_results(self, sensor_type=None):
        """Get detection results"""
        if sensor_type:
            return self.results.get(sensor_type, {})
        return self.results