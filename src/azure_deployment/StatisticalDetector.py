import pandas as pd
import numpy as np
import logging
import time
from tqdm import tqdm

class StatisticalDetector:
    """Statistical anomaly detection using robust statistical methods.
    
    This detector uses:
    1. Threshold violation detection (highest priority)
    2. Z-score analysis with adaptive thresholds
    3. Rolling statistics for trend analysis
    
    It's designed to be memory-efficient and handle large datasets through
    batch processing and sensor-by-sensor analysis.
    """
    
    def __init__(self, config=None, logger=None):
        """Initialize the statistical detector"""
        self.default_config = {
            'temperature': {
                'z_score_threshold': 3.5,  # Higher threshold to reduce false positives
                'rolling_window': 12,     # 1 hour for 5-min data
                'batch_size': 50000,
                'adaptive_params': {
                    'sensitivity': 'low',  # Low sensitivity to avoid false positives
                    'seasonality_adjustment': True,
                },
                'min_data_points': 10
            },
            'battery': {
                'z_score_threshold': 4.0,   # Higher threshold for battery (less variation expected)
                'rolling_window': 8,       # 2 hours for 15-min data
                'batch_size': 30000,
                'adaptive_params': {
                    'sensitivity': 'low',  # Lower sensitivity to reduce false alarms
                    'trend_consideration': True,
                },
                'min_data_points': 8,
                'min_voltage_threshold': 2.0  # Minimum voltage threshold for batteries
            },
            'vibration': {
                'z_score_threshold': 4.0,   # Higher threshold to reduce false positives
                'rolling_window': 6,       # 6 hours for hourly data
                'batch_size': 10000,
                'adaptive_params': {
                    'sensitivity': 'low',
                    'magnitude_variation': True,
                },
                'min_data_points': 6,
                'use_magnitude_rms': True  # Use RMS magnitude for vibration
            },
            'motion': {
                'z_score_threshold': 4.0,
                'rolling_window': 7,       # 7 days for daily data
                'batch_size': 5000,
                'adaptive_params': {
                    'sensitivity': 'low',
                    'event_clustering': True,
                },
                'min_data_points': 5,
                'all_motion_as_anomaly': True  # All motion events are treated as anomalies
            }
        }
        
        self.config = config or self.default_config
        self.logger = logger or logging.getLogger('statistical_detector')
        self.results = {}
        
    def calibrate_threshold(self, sensor_data, sensor_type, value_col):
        """Dynamically calibrate anomaly detection thresholds based on data characteristics"""
        if not isinstance(sensor_data, pd.DataFrame) or value_col not in sensor_data.columns:
            # Handle invalid input
            return {'z_score_threshold': self.config[sensor_type]['z_score_threshold']}
            
        config = self.config.get(sensor_type, self.config['temperature'])
        adaptive_params = config.get('adaptive_params', {})
        
        if len(sensor_data) < adaptive_params.get('min_data_points', 10):
            return {'z_score_threshold': config['z_score_threshold']}
        
        try:
            # Calculate data characteristics
            data = sensor_data[value_col].dropna()
            
            # Skip if not enough valid data
            if len(data) < 5:
                return {'z_score_threshold': config['z_score_threshold']}
                
            data_range = data.max() - data.min()
            data_std = data.std()
            data_mean = data.mean()
            
            if data_std == 0 or data_range == 0:
                return {'z_score_threshold': config['z_score_threshold']}
            
            # Coefficient of variation (normalized measure of dispersion)
            cv = data_std / abs(data_mean) if data_mean != 0 else float('inf')
            
            sensitivity_map = {
                'low': 1.5,     # More tolerant (fewer false positives)
                'medium': 1.0,  # Default
                'high': 0.75,   # More sensitive
                'adaptive': 1.0 * min(1.0, cv)  # Adapt based on coefficient of variation
            }
            
            sensitivity = adaptive_params.get('sensitivity', 'low')
            sensitivity_factor = sensitivity_map.get(sensitivity, 1.5)
            
            z_threshold = config['z_score_threshold'] * sensitivity_factor
            
            # Check for seasonality if enabled and we have sufficient data
            if adaptive_params.get('seasonality_adjustment', False) and len(sensor_data) >= 24:
                if 'event_time' in sensor_data.columns:
                    try:
                        # Make sure event_time is datetime
                        if not pd.api.types.is_datetime64_any_dtype(sensor_data['event_time']):
                            sensor_data['event_time'] = pd.to_datetime(sensor_data['event_time'])
                            
                        hourly_means = sensor_data.groupby(sensor_data['event_time'].dt.hour)[value_col].mean()
                        hourly_std = hourly_means.std()
                        
                        if hourly_std > 0:
                            # Calculate seasonal strength (ratio of hourly variation to overall variation)
                            seasonality_strength = hourly_std / data_std
                            
                            if seasonality_strength > 0.3:  # Threshold for significant seasonality
                                z_threshold *= (1 + min(1.0, seasonality_strength))
                                
                                # Check if this is a recurring seasonal pattern
                                if len(sensor_data) >= 48:  # At least 2 days of data
                                    try:
                                        day_hour_groups = sensor_data.groupby([
                                            sensor_data['event_time'].dt.day, 
                                            sensor_data['event_time'].dt.hour
                                        ])[value_col].mean()
                                        
                                        if len(day_hour_groups) > 0:
                                            try:
                                                pivot_data = day_hour_groups.unstack()
                                                
                                                # Only proceed if we have enough days
                                                if len(pivot_data) >= 2:
                                                    corr_matrix = pivot_data.T.corr()
                                                    
                                                    # Average correlation excluding self-correlations
                                                    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
                                                    
                                                    if avg_corr > 0.7:  # Strong correlation threshold
                                                        z_threshold *= 1.2  # 20% higher threshold for confirmed seasonality
                                            except:
                                                pass  # Skip correlation calculation if it fails
                                    except:
                                        pass  # Skip if groupby fails
                    except Exception as e:
                        self.logger.debug(f"Error in seasonality adjustment: {str(e)}")
            
            # For battery data, ensure we respect minimum voltage threshold
            if sensor_type == 'battery' and 'min_voltage_threshold' in config:
                min_voltage = config['min_voltage_threshold']
                result = {
                    'z_score_threshold': z_threshold,
                    'min_voltage_threshold': min_voltage,
                    'data_range': float(data_range),
                    'data_std': float(data_std),
                    'data_mean': float(data_mean),
                    'coefficient_of_variation': float(cv)
                }
                return result
            
            # For motion data, if configured to treat all motion as anomaly
            if sensor_type == 'motion' and config.get('all_motion_as_anomaly', False):
                result = {
                    'z_score_threshold': z_threshold,
                    'all_as_anomaly': True,
                    'data_range': float(data_range),
                    'data_std': float(data_std),
                    'data_mean': float(data_mean),
                    'coefficient_of_variation': float(cv)
                }
                return result
                
            # For vibration data, if configured to use RMS magnitude
            if sensor_type == 'vibration' and config.get('use_magnitude_rms', False):
                result = {
                    'z_score_threshold': z_threshold,
                    'use_magnitude_rms': True,
                    'data_range': float(data_range),
                    'data_std': float(data_std),
                    'data_mean': float(data_mean),
                    'coefficient_of_variation': float(cv)
                }
                return result
                
            result = {
                'z_score_threshold': z_threshold,
                'data_range': float(data_range),
                'data_std': float(data_std),
                'data_mean': float(data_mean),
                'coefficient_of_variation': float(cv)
            }
            return result
            
        except Exception as e:
            self.logger.error(f"Error in threshold calibration: {str(e)}")
            return {'z_score_threshold': config['z_score_threshold']}
            
    def detect(self, df, sensor_type):
        """Perform statistical anomaly detection"""
        if df is None or df.empty:
            self.logger.warning(f"Empty dataframe for {sensor_type}, skipping detection")
            return df
        
        self.logger.info(f"Starting statistical anomaly detection for {sensor_type} data")
        
        try:
            # Validate input columns
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
            
            # Get value column and threshold columns
            value_col, threshold_cols = self.get_columns(df, sensor_type)
            if not value_col:
                self.logger.error(f"Could not determine value column for {sensor_type}")
                return df
                
            # Add statistical anomaly columns
            df['statistical_anomaly'] = False
            df['threshold_violation'] = False
            df['z_score'] = np.nan
            
            # Get configuration for this sensor type
            config = self.config.get(sensor_type, self.config['temperature'])
            
            # Special handling for motion data if configured to treat all motion as anomalies
            if sensor_type == 'motion' and config.get('all_motion_as_anomaly', True):
                self.logger.info("Treating all motion events as anomalies")
                df['statistical_anomaly'] = True
                
                anomaly_count = len(df)
                self.logger.info(f"Statistical detection results for {sensor_type}: {anomaly_count} statistical anomalies, {anomaly_count/len(df)*100:.2f}%")
                
                self.results[sensor_type] = {
                    'anomaly_count': int(anomaly_count),
                    'statistical_anomaly_count': int(anomaly_count),
                    'threshold_violation_count': 0,
                    'total_records': len(df),
                    'anomaly_percentage': float(anomaly_count/len(df)*100)
                }
                return df
                
            # Special handling for battery data to enforce minimum voltage threshold
            if sensor_type == 'battery' and 'min_voltage_threshold' in config and value_col in df.columns:
                min_voltage = config['min_voltage_threshold']
                # Mark readings below minimum voltage as threshold violations
                battery_threshold_violations = df[value_col] < min_voltage
                df.loc[battery_threshold_violations, 'threshold_violation'] = True
                self.logger.info(f"Identified {battery_threshold_violations.sum()} battery readings below minimum threshold of {min_voltage}V")
            
            # Process each sensor individually
            for sensor_id in tqdm(df['sensor_id'].unique(), desc=f"Statistical analysis {sensor_type}"):
                sensor_df = df[df['sensor_id'] == sensor_id]
                
                # Skip if not enough data points
                if len(sensor_df) < config.get('min_data_points', 5) or value_col not in sensor_df.columns:
                    continue
                    
                # Calculate z-scores for this sensor
                try:
                    mean = sensor_df[value_col].mean()
                    std = sensor_df[value_col].std()
                    
                    if std > 0:
                        # Calculate z-scores
                        z_scores = (sensor_df[value_col] - mean) / std
                        
                        # Get calibrated threshold for this sensor
                        try:
                            calibrated = self.calibrate_threshold(sensor_df, sensor_type, value_col)
                            z_threshold = calibrated.get('z_score_threshold', config['z_score_threshold'])
                        except Exception as e:
                            self.logger.warning(f"Error calibrating threshold for {sensor_id}: {str(e)}")
                            z_threshold = config['z_score_threshold']
                        
                        # Mark anomalies based on z-score
                        anomalies = abs(z_scores) > z_threshold
                        df.loc[sensor_df.index[anomalies], 'statistical_anomaly'] = True
                        df.loc[sensor_df.index, 'z_score'] = z_scores
                except Exception as e:
                    self.logger.warning(f"Error calculating z-scores for {sensor_id}: {str(e)}")
            
            # Count anomalies
            anomaly_count = df['statistical_anomaly'].sum()
            threshold_count = df['threshold_violation'].sum()
            total_anomalies = anomaly_count + threshold_count
            
            self.logger.info(f"Statistical detection results for {sensor_type}: {total_anomalies} anomalies ({anomaly_count} statistical, {threshold_count} threshold), {total_anomalies/len(df)*100:.2f}%")
            
            self.results[sensor_type] = {
                'anomaly_count': int(total_anomalies),
                'statistical_anomaly_count': int(anomaly_count),
                'threshold_violation_count': int(threshold_count),
                'total_records': len(df),
                'anomaly_percentage': float(total_anomalies/len(df)*100),
                'value_column': value_col
            }
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in statistical detection for {sensor_type}: {str(e)}")
            # Return the original dataframe to allow the pipeline to continue
            return df
        
    def get_columns(self, df, sensor_type):
        """Get the value column and threshold columns for a sensor type"""
        if df is None or df.empty:
            return None, []
            
        if sensor_type == 'temperature':
            value_col = 'temperature_celsius'
            threshold_cols = ['high_threshold', 'low_threshold']
            
            # Handle original column names if needed
            if value_col not in df.columns and 'TemperatureInCelsius' in df.columns:
                value_col = 'TemperatureInCelsius'
            if value_col not in df.columns:
                temp_cols = [col for col in df.columns if 'temperature' in col.lower() or 'temp' in col.lower()]
                if temp_cols:
                    value_col = temp_cols[0]
                    
        elif sensor_type == 'battery':
            value_col = 'battery_voltage'
            threshold_cols = ['low_threshold']
            
            # Handle original column names if needed
            if value_col not in df.columns and 'BatteryVoltage' in df.columns:
                value_col = 'BatteryVoltage'
            if value_col not in df.columns:
                batt_cols = [col for col in df.columns if 'battery' in col.lower() or 'voltage' in col.lower()]
                if batt_cols:
                    value_col = batt_cols[0]
                    
        elif sensor_type == 'vibration':
            value_col = 'vibration_magnitude'
            threshold_cols = ['high_threshold']
            
            # Handle original column names if needed
            if value_col not in df.columns and 'MagnitudeRMSInMilliG' in df.columns:
                value_col = 'MagnitudeRMSInMilliG'
            if value_col not in df.columns:
                # Look for any magnitude columns
                magnitude_cols = [col for col in df.columns if 'magnitude' in col.lower() or 'rms' in col.lower()]
                if magnitude_cols:
                    value_col = magnitude_cols[0]
                    
        elif sensor_type == 'motion':
            value_col = None
            threshold_cols = []
            
            # Try to find appropriate columns
            if 'max_magnitude' in df.columns:
                value_col = 'max_magnitude'
            elif 'daily_event_count' in df.columns:
                value_col = 'daily_event_count'
            
            # If no specific motion columns found, look for any event-related columns
            if not value_col:
                event_cols = [col for col in df.columns if 'event' in col.lower() or 'count' in col.lower()]
                if event_cols:
                    value_col = event_cols[0]
                    
        else:
            value_col = None
            threshold_cols = []
            
        # Filter threshold columns to only those that exist
        threshold_cols = [col for col in threshold_cols if col in df.columns]
        
        # Look for original threshold column names if none found
        if not threshold_cols:
            threshold_cols = [col for col in df.columns if 'threshold' in col.lower()]
        
        # Check if value column actually exists
        if value_col and value_col not in df.columns:
            self.logger.warning(f"Value column {value_col} not found in {sensor_type} data")
            return None, threshold_cols
            
        return value_col, threshold_cols
        
    def get_results(self, sensor_type=None):
        """Get detection results"""
        if sensor_type:
            return self.results.get(sensor_type, {})
        return self.results