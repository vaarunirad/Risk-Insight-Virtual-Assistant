import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta

class SensorPreprocessor:
    """Preprocess sensor data for anomaly detection.
       This class handles:
    1. Resampling data to appropriate intervals
    2. Handling missing values
    3. Outlier removal
    4. Feature engineering
    5. Data validation
    """

    def __init__(self, config=None, logger=None):
        """Initialize the sensor preprocessor"""
        self.default_config = {
            'temperature': {
                'resample_freq': '5min',    # 5-minute resampling
                'interpolation': 'linear',  # Linear interpolation for missing values
                'outlier_removal': True,    # Remove outliers
                'outlier_threshold': 5.0,   # Z-score threshold for outliers
                'max_interpolation_gap': 6  # Maximum gap (in intervals) to interpolate
            },
            'battery': {
                'resample_freq': '15min',   # 15-minute resampling
                'interpolation': 'linear',  # Linear interpolation for missing values
                'outlier_removal': True,    # Remove outliers
                'outlier_threshold': 5.0,   # Z-score threshold for outliers
                'max_interpolation_gap': 4  # Maximum gap (in intervals) to interpolate
            },
            'vibration': {
                'resample_freq': '1H',      # Hourly resampling
                'interpolation': 'linear',  # Linear interpolation for missing values
                'outlier_removal': True,    # Remove outliers
                'outlier_threshold': 5.0,   # Z-score threshold for outliers
                'max_interpolation_gap': 3  # Maximum gap (in intervals) to interpolate
            },
            'motion': {
                'resample_freq': '1D',      # Daily resampling
                'aggregation': 'max',       # Aggregation method for motion
                'event_counting': True,     # Count motion events
                'outlier_removal': False,   # Don't remove outliers for motion
                'max_interpolation_gap': 0  # Don't interpolate motion data
            },
            'memory_optimization': {
                'chunk_size': 100000,       # Process in chunks
                'use_dask': False,          # Use Dask for large datasets
                'optimize_dtypes': True     # Optimize data types to reduce memory
            }
        }
            
        self.config = config or self.default_config
        self.logger = logger or logging.getLogger('sensor_preprocessor')
            
    def preprocess_temperature(self, df):
        """Preprocess temperature sensor data"""
        if df is None or df.empty:
            self.logger.warning("Empty temperature dataframe, skipping preprocessing")
            return df
                
        self.logger.info("Preprocessing temperature data")
            
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
            
        # Validate input data to make sure required columns exist
        required_cols = ['SensorId', 'EventTimeUtc', 'TemperatureInCelsius']
        missing_cols = [col for col in required_cols if col not in df_copy.columns]
        
        if missing_cols:
            self.logger.warning(f"Missing required columns for temperature data: {missing_cols}")
            # Return original data if we can't process it
            return df
            
        # Process each sensor separately with robust error handling
        result_dfs = []
        
        try:
            for sensor_id in df_copy['SensorId'].unique():
                # Get sensor data
                try:
                    sensor_df = df_copy[df_copy['SensorId'] == sensor_id].copy()
                    
                    # Skip if not enough data
                    if len(sensor_df) < 5:
                        self.logger.debug(f"Sensor {sensor_id} has insufficient data points ({len(sensor_df)}), skipping resampling")
                        result_dfs.append(sensor_df)
                        continue
                    
                    # Process this sensor
                    processed_df = self._process_temperature_sensor(sensor_df, sensor_id)
                    result_dfs.append(processed_df)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing temperature sensor {sensor_id}: {str(e)}")
                    # Add the original sensor data to prevent data loss
                    sensor_df = df_copy[df_copy['SensorId'] == sensor_id]
                    if not sensor_df.empty:
                        result_dfs.append(sensor_df)
            
            if result_dfs:
                # Combine all results
                result = pd.concat(result_dfs, ignore_index=True)
                return result
            else:
                self.logger.warning("No temperature data could be processed")
                return df_copy
                
        except Exception as e:
            self.logger.error(f"Error in temperature preprocessing: {str(e)}")
            # Return original data on error
            return df_copy
    
    def _process_temperature_sensor(self, sensor_df, sensor_id):
        """Process a single temperature sensor with robust error handling"""
        # Get configuration
        config = self.config['temperature']
        resample_freq = config.get('resample_freq', '5min')
        interpolation = config.get('interpolation', 'linear')
        outlier_removal = config.get('outlier_removal', True)
        outlier_threshold = config.get('outlier_threshold', 5.0)
        max_gap = config.get('max_interpolation_gap', 6)
        
        try:
            # Ensure event time is datetime
            if 'EventTimeUtc' in sensor_df.columns:
                if not pd.api.types.is_datetime64_any_dtype(sensor_df['EventTimeUtc']):
                    sensor_df['EventTimeUtc'] = pd.to_datetime(sensor_df['EventTimeUtc'], errors='coerce')
                    # Drop rows with invalid timestamps
                    sensor_df = sensor_df.dropna(subset=['EventTimeUtc'])
                    
                    if sensor_df.empty:
                        self.logger.warning(f"No valid timestamps for sensor {sensor_id} after conversion")
                        return sensor_df
                
                # Set time as index for resampling
                sensor_df.set_index('EventTimeUtc', inplace=True)
            else:
                self.logger.warning(f"No EventTimeUtc column found for sensor {sensor_id}")
                # If no timestamp, return original data
                return sensor_df.reset_index() if hasattr(sensor_df, 'reset_index') else sensor_df
            
            # Remove outliers if configured
            if outlier_removal and 'TemperatureInCelsius' in sensor_df.columns:
                try:
                    mean = sensor_df['TemperatureInCelsius'].mean()
                    std = sensor_df['TemperatureInCelsius'].std()
                    
                    if std > 0:
                        z_scores = abs((sensor_df['TemperatureInCelsius'] - mean) / std)
                        sensor_df = sensor_df[z_scores <= outlier_threshold]
                except Exception as e:
                    self.logger.debug(f"Error removing outliers for sensor {sensor_id}: {str(e)}")
            
            # Create aggregation dictionary for robust resampling
            agg_dict = {}
            
            # Handle numeric columns
            for col in sensor_df.select_dtypes(include=['float', 'int']).columns:
                agg_dict[col] = 'mean'
                
            # Handle categorical/object columns
            for col in sensor_df.select_dtypes(exclude=['float', 'int']).columns:
                agg_dict[col] = 'first'
                
            # Perform resampling with aggregation
            try:
                resampled_df = sensor_df.resample(resample_freq).agg(agg_dict)
                
                # Interpolate missing values in temperature
                if 'TemperatureInCelsius' in resampled_df.columns:
                    resampled_df['TemperatureInCelsius'] = resampled_df['TemperatureInCelsius'].interpolate(
                        method=interpolation,
                        limit=max_gap,
                        limit_direction='both'
                    )
                
                # Forward fill categorical columns
                for col in sensor_df.select_dtypes(exclude=['float', 'int']).columns:
                    if col in resampled_df.columns:
                        resampled_df[col] = resampled_df[col].ffill()
                
                # Reset index to convert back to regular column
                resampled_df = resampled_df.reset_index()
                
                # Add back sensor ID if it was lost
                if 'SensorId' not in resampled_df.columns:
                    resampled_df['SensorId'] = sensor_id
                    
                return resampled_df
                
            except Exception as e:
                self.logger.warning(f"Error during resampling for sensor {sensor_id}: {str(e)}")
                # If resampling fails, return original data with index reset
                return sensor_df.reset_index()
                
        except Exception as e:
            self.logger.error(f"Error in _process_temperature_sensor for {sensor_id}: {str(e)}")
            # If any unexpected error, return original data 
            if hasattr(sensor_df, 'reset_index'):
                # If sensor_df has index, reset it to ensure EventTimeUtc is a column
                if sensor_df.index.name == 'EventTimeUtc':
                    return sensor_df.reset_index()
            
            # Otherwise return as is
            return sensor_df
                
    def preprocess_battery(self, df):
        """Preprocess battery sensor data"""
        if df is None or df.empty:
            self.logger.warning("Empty battery dataframe, skipping preprocessing")
            return df
                
        self.logger.info("Preprocessing battery data")
            
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
            
        # Validate input data to make sure required columns exist
        required_cols = ['SensorId', 'EventTimeUtc', 'BatteryVoltage']
        missing_cols = [col for col in required_cols if col not in df_copy.columns]
        
        if missing_cols:
            self.logger.warning(f"Missing required columns for battery data: {missing_cols}")
            # Return original data if we can't process it
            return df
            
        # Process each sensor separately with robust error handling
        result_dfs = []
        
        try:
            for sensor_id in df_copy['SensorId'].unique():
                # Get sensor data
                try:
                    sensor_df = df_copy[df_copy['SensorId'] == sensor_id].copy()
                    
                    # Skip if not enough data
                    if len(sensor_df) < 5:
                        self.logger.debug(f"Sensor {sensor_id} has insufficient data points ({len(sensor_df)}), skipping resampling")
                        result_dfs.append(sensor_df)
                        continue
                    
                    # Process this sensor
                    processed_df = self._process_battery_sensor(sensor_df, sensor_id)
                    result_dfs.append(processed_df)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing battery sensor {sensor_id}: {str(e)}")
                    # Add the original sensor data to prevent data loss
                    sensor_df = df_copy[df_copy['SensorId'] == sensor_id]
                    if not sensor_df.empty:
                        result_dfs.append(sensor_df)
            
            if result_dfs:
                # Combine all results
                result = pd.concat(result_dfs, ignore_index=True)
                return result
            else:
                self.logger.warning("No battery data could be processed")
                return df_copy
                
        except Exception as e:
            self.logger.error(f"Error in battery preprocessing: {str(e)}")
            # Return original data on error
            return df_copy
    
    def _process_battery_sensor(self, sensor_df, sensor_id):
        """Process a single battery sensor with robust error handling"""
        # Get configuration
        config = self.config['battery']
        resample_freq = config.get('resample_freq', '15min')
        interpolation = config.get('interpolation', 'linear')
        outlier_removal = config.get('outlier_removal', True)
        outlier_threshold = config.get('outlier_threshold', 5.0)
        max_gap = config.get('max_interpolation_gap', 4)
        
        try:
            # Ensure event time is datetime
            if 'EventTimeUtc' in sensor_df.columns:
                if not pd.api.types.is_datetime64_any_dtype(sensor_df['EventTimeUtc']):
                    sensor_df['EventTimeUtc'] = pd.to_datetime(sensor_df['EventTimeUtc'], errors='coerce')
                    # Drop rows with invalid timestamps
                    sensor_df = sensor_df.dropna(subset=['EventTimeUtc'])
                    
                    if sensor_df.empty:
                        self.logger.warning(f"No valid timestamps for sensor {sensor_id} after conversion")
                        return sensor_df
                
                # Set time as index for resampling
                sensor_df.set_index('EventTimeUtc', inplace=True)
            else:
                self.logger.warning(f"No EventTimeUtc column found for sensor {sensor_id}")
                # If no timestamp, return original data
                return sensor_df.reset_index() if hasattr(sensor_df, 'reset_index') else sensor_df
            
            # Remove outliers if configured
            if outlier_removal and 'BatteryVoltage' in sensor_df.columns:
                try:
                    mean = sensor_df['BatteryVoltage'].mean()
                    std = sensor_df['BatteryVoltage'].std()
                    
                    if std > 0:
                        z_scores = abs((sensor_df['BatteryVoltage'] - mean) / std)
                        sensor_df = sensor_df[z_scores <= outlier_threshold]
                except Exception as e:
                    self.logger.debug(f"Error removing outliers for sensor {sensor_id}: {str(e)}")
            
            # Create aggregation dictionary for robust resampling
            agg_dict = {}
            
            # Handle numeric columns
            for col in sensor_df.select_dtypes(include=['float', 'int']).columns:
                agg_dict[col] = 'mean'
                
            # Handle categorical/object columns
            for col in sensor_df.select_dtypes(exclude=['float', 'int']).columns:
                agg_dict[col] = 'first'
                
            # Perform resampling with aggregation
            try:
                resampled_df = sensor_df.resample(resample_freq).agg(agg_dict)
                
                # Interpolate missing values in battery voltage
                if 'BatteryVoltage' in resampled_df.columns:
                    resampled_df['BatteryVoltage'] = resampled_df['BatteryVoltage'].interpolate(
                        method=interpolation,
                        limit=max_gap,
                        limit_direction='both'
                    )
                
                # Forward fill categorical columns
                for col in sensor_df.select_dtypes(exclude=['float', 'int']).columns:
                    if col in resampled_df.columns:
                        resampled_df[col] = resampled_df[col].ffill()
                
                # Reset index to convert back to regular column
                resampled_df = resampled_df.reset_index()
                
                # Add back sensor ID if it was lost
                if 'SensorId' not in resampled_df.columns:
                    resampled_df['SensorId'] = sensor_id
                    
                return resampled_df
                
            except Exception as e:
                self.logger.warning(f"Error during resampling for sensor {sensor_id}: {str(e)}")
                # If resampling fails, return original data with index reset
                return sensor_df.reset_index()
                
        except Exception as e:
            self.logger.error(f"Error in _process_battery_sensor for {sensor_id}: {str(e)}")
            # If any unexpected error, return original data 
            if hasattr(sensor_df, 'reset_index'):
                # If sensor_df has index, reset it to ensure EventTimeUtc is a column
                if sensor_df.index.name == 'EventTimeUtc':
                    return sensor_df.reset_index()
            
            # Otherwise return as is
            return sensor_df
                
    def preprocess_vibration(self, df):
        """Preprocess vibration sensor data"""
        if df is None or df.empty:
            self.logger.warning("Empty vibration dataframe, skipping preprocessing")
            return df
                
        self.logger.info("Preprocessing vibration data")
            
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
            
        # Validate input data to make sure required columns exist
        required_cols = ['SensorId', 'EventTimeUtc']
        missing_cols = [col for col in required_cols if col not in df_copy.columns]
        
        if missing_cols:
            self.logger.warning(f"Missing required columns for vibration data: {missing_cols}")
            # Return original data if we can't process it
            return df
            
        # Process each sensor separately with robust error handling
        result_dfs = []
        
        try:
            for sensor_id in df_copy['SensorId'].unique():
                # Get sensor data
                try:
                    sensor_df = df_copy[df_copy['SensorId'] == sensor_id].copy()
                    
                    # Skip if not enough data
                    if len(sensor_df) < 5:
                        self.logger.debug(f"Sensor {sensor_id} has insufficient data points ({len(sensor_df)}), skipping resampling")
                        result_dfs.append(sensor_df)
                        continue
                    
                    # Process this sensor
                    processed_df = self._process_vibration_sensor(sensor_df, sensor_id)
                    result_dfs.append(processed_df)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing vibration sensor {sensor_id}: {str(e)}")
                    # Add the original sensor data to prevent data loss
                    sensor_df = df_copy[df_copy['SensorId'] == sensor_id]
                    if not sensor_df.empty:
                        result_dfs.append(sensor_df)
            
            if result_dfs:
                # Combine all results
                result = pd.concat(result_dfs, ignore_index=True)
                return result
            else:
                self.logger.warning("No vibration data could be processed")
                return df_copy
                
        except Exception as e:
            self.logger.error(f"Error in vibration preprocessing: {str(e)}")
            # Return original data on error
            return df_copy
    
    def _process_vibration_sensor(self, sensor_df, sensor_id):
        """Process a single vibration sensor with robust error handling"""
        # Get configuration
        config = self.config['vibration']
        resample_freq = config.get('resample_freq', '1H')
        interpolation = config.get('interpolation', 'linear')
        outlier_removal = config.get('outlier_removal', True)
        outlier_threshold = config.get('outlier_threshold', 5.0)
        max_gap = config.get('max_interpolation_gap', 3)
        
        try:
            # Ensure event time is datetime
            if 'EventTimeUtc' in sensor_df.columns:
                if not pd.api.types.is_datetime64_any_dtype(sensor_df['EventTimeUtc']):
                    sensor_df['EventTimeUtc'] = pd.to_datetime(sensor_df['EventTimeUtc'], errors='coerce')
                    # Drop rows with invalid timestamps
                    sensor_df = sensor_df.dropna(subset=['EventTimeUtc'])
                    
                    if sensor_df.empty:
                        self.logger.warning(f"No valid timestamps for sensor {sensor_id} after conversion")
                        return sensor_df
                
                # Set time as index for resampling
                sensor_df.set_index('EventTimeUtc', inplace=True)
            else:
                self.logger.warning(f"No EventTimeUtc column found for sensor {sensor_id}")
                # If no timestamp, return original data
                return sensor_df.reset_index() if hasattr(sensor_df, 'reset_index') else sensor_df
            
            # Look for magnitude columns
            magnitude_cols = [col for col in sensor_df.columns if 'Magnitude' in col and 'Threshold' not in col]
            
            # Remove outliers if configured and magnitude columns exist
            if outlier_removal and magnitude_cols:
                main_magnitude_col = magnitude_cols[0]  # Use the first magnitude column
                try:
                    mean = sensor_df[main_magnitude_col].mean()
                    std = sensor_df[main_magnitude_col].std()
                    
                    if std > 0:
                        z_scores = abs((sensor_df[main_magnitude_col] - mean) / std)
                        sensor_df = sensor_df[z_scores <= outlier_threshold]
                except Exception as e:
                    self.logger.debug(f"Error removing outliers for sensor {sensor_id}: {str(e)}")
            
            # Create aggregation dictionary for robust resampling
            agg_dict = {}
            
            # Handle numeric columns
            for col in sensor_df.select_dtypes(include=['float', 'int']).columns:
                if 'Magnitude' in col:
                    # Use max for magnitude to preserve burst patterns
                    agg_dict[col] = 'max'
                else:
                    agg_dict[col] = 'mean'
                
            # Handle categorical/object columns
            for col in sensor_df.select_dtypes(exclude=['float', 'int']).columns:
                agg_dict[col] = 'first'
                
            # Perform resampling with aggregation
            try:
                resampled_df = sensor_df.resample(resample_freq).agg(agg_dict)
                
                # Interpolate missing values in magnitude columns
                for col in magnitude_cols:
                    if col in resampled_df.columns:
                        resampled_df[col] = resampled_df[col].interpolate(
                            method=interpolation,
                            limit=max_gap,
                            limit_direction='both'
                        )
                
                # Forward fill categorical columns
                for col in sensor_df.select_dtypes(exclude=['float', 'int']).columns:
                    if col in resampled_df.columns:
                        resampled_df[col] = resampled_df[col].ffill()
                
                # Reset index to convert back to regular column
                resampled_df = resampled_df.reset_index()
                
                # Add back sensor ID if it was lost
                if 'SensorId' not in resampled_df.columns:
                    resampled_df['SensorId'] = sensor_id
                    
                return resampled_df
                
            except Exception as e:
                self.logger.warning(f"Error during resampling for sensor {sensor_id}: {str(e)}")
                # If resampling fails, return original data with index reset
                return sensor_df.reset_index()
                
        except Exception as e:
            self.logger.error(f"Error in _process_vibration_sensor for {sensor_id}: {str(e)}")
            # If any unexpected error, return original data 
            if hasattr(sensor_df, 'reset_index'):
                # If sensor_df has index, reset it to ensure EventTimeUtc is a column
                if sensor_df.index.name == 'EventTimeUtc':
                    return sensor_df.reset_index()
            
            # Otherwise return as is
            return sensor_df
                
    def preprocess_motion(self, df):
        """Preprocess motion sensor data"""
        if df is None or df.empty:
            self.logger.warning("Empty motion dataframe, skipping preprocessing")
            return df
                
        self.logger.info("Preprocessing motion data")
            
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
            
        # Validate input data to make sure required columns exist
        required_cols = ['SensorId', 'EventTimeUtc']
        missing_cols = [col for col in required_cols if col not in df_copy.columns]
        
        if missing_cols:
            self.logger.warning(f"Missing required columns for motion data: {missing_cols}")
            # Return original data if we can't process it
            return df
            
        # Process each sensor separately with robust error handling
        result_dfs = []
        
        try:
            for sensor_id in df_copy['SensorId'].unique():
                # Get sensor data
                try:
                    sensor_df = df_copy[df_copy['SensorId'] == sensor_id].copy()
                    
                    # Skip if not enough data
                    if len(sensor_df) < 2:  # Need at least 2 points for motion
                        self.logger.debug(f"Sensor {sensor_id} has insufficient data points ({len(sensor_df)}), skipping resampling")
                        result_dfs.append(sensor_df)
                        continue
                    
                    # Process this sensor
                    processed_df = self._process_motion_sensor(sensor_df, sensor_id)
                    result_dfs.append(processed_df)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing motion sensor {sensor_id}: {str(e)}")
                    # Add the original sensor data to prevent data loss
                    sensor_df = df_copy[df_copy['SensorId'] == sensor_id]
                    if not sensor_df.empty:
                        result_dfs.append(sensor_df)
            
            if result_dfs:
                # Combine all results
                result = pd.concat(result_dfs, ignore_index=True)
                return result
            else:
                self.logger.warning("No motion data could be processed")
                return df_copy
                
        except Exception as e:
            self.logger.error(f"Error in motion preprocessing: {str(e)}")
            # Return original data on error
            return df_copy
    
    def _process_motion_sensor(self, sensor_df, sensor_id):
        """Process a single motion sensor with robust error handling"""
        # Get configuration
        config = self.config['motion']
        resample_freq = config.get('resample_freq', '1D')
        aggregation = config.get('aggregation', 'max')
        event_counting = config.get('event_counting', True)
        
        try:
            # Ensure event time is datetime
            if 'EventTimeUtc' in sensor_df.columns:
                if not pd.api.types.is_datetime64_any_dtype(sensor_df['EventTimeUtc']):
                    sensor_df['EventTimeUtc'] = pd.to_datetime(sensor_df['EventTimeUtc'], errors='coerce')
                    # Drop rows with invalid timestamps
                    sensor_df = sensor_df.dropna(subset=['EventTimeUtc'])
                    
                    if sensor_df.empty:
                        self.logger.warning(f"No valid timestamps for sensor {sensor_id} after conversion")
                        return sensor_df
                
                # Set time as index for resampling
                sensor_df.set_index('EventTimeUtc', inplace=True)
            else:
                self.logger.warning(f"No EventTimeUtc column found for sensor {sensor_id}")
                # If no timestamp, return original data
                return sensor_df.reset_index() if hasattr(sensor_df, 'reset_index') else sensor_df
            
            # For motion data, we treat it as event-based
            if event_counting:
                try:
                    # Count events per day
                    daily_counts = sensor_df.resample(resample_freq).size()
                    
                    # Convert to dataframe
                    df_counts = pd.DataFrame(daily_counts, columns=['daily_event_count'])
                    
                    # Add back sensor ID
                    df_counts['SensorId'] = sensor_id
                    
                    # Look for magnitude columns
                    magnitude_cols = [col for col in sensor_df.columns if 'Magnitude' in col]
                    
                    # Calculate max of magnitude columns if they exist
                    for col in magnitude_cols:
                        try:
                            df_counts[f'max_{col}'] = sensor_df[col].resample(resample_freq).max()
                        except Exception as e:
                            self.logger.debug(f"Error calculating max for {col}: {str(e)}")
                    
                    # Reset index
                    df_counts = df_counts.reset_index()
                    
                    return df_counts
                    
                except Exception as e:
                    self.logger.warning(f"Error in event counting for sensor {sensor_id}: {str(e)}")
                    # Fall back to simple resampling
                    
            # Create aggregation dictionary for resampling
            agg_dict = {}
            
            # Handle numeric columns
            for col in sensor_df.select_dtypes(include=['float', 'int']).columns:
                agg_dict[col] = aggregation  # Use the specified aggregation method
                
            # Handle categorical/object columns
            for col in sensor_df.select_dtypes(exclude=['float', 'int']).columns:
                agg_dict[col] = 'first'
                
            # Perform resampling with aggregation
            try:
                resampled_df = sensor_df.resample(resample_freq).agg(agg_dict)
                
                # Reset index to convert back to regular column
                resampled_df = resampled_df.reset_index()
                
                # Add back sensor ID if it was lost
                if 'SensorId' not in resampled_df.columns:
                    resampled_df['SensorId'] = sensor_id
                    
                return resampled_df
                
            except Exception as e:
                self.logger.warning(f"Error during resampling for sensor {sensor_id}: {str(e)}")
                # If resampling fails, return original data with index reset
                return sensor_df.reset_index()
                
        except Exception as e:
            self.logger.error(f"Error in _process_motion_sensor for {sensor_id}: {str(e)}")
            # If any unexpected error, return original data 
            if hasattr(sensor_df, 'reset_index'):
                # If sensor_df has index, reset it to ensure EventTimeUtc is a column
                if sensor_df.index.name == 'EventTimeUtc':
                    return sensor_df.reset_index()
            
            # Otherwise return as is
            return sensor_df
                
    def validate_data(self, df, sensor_type):
        """Validate data for a specific sensor type"""
        if df is None or df.empty:
            return False, "Empty dataframe"
                
        # Check required columns
        required_columns = ['SensorId', 'EventTimeUtc']
            
        if sensor_type == 'temperature':
            required_columns.append('TemperatureInCelsius')
        elif sensor_type == 'battery':
            required_columns.append('BatteryVoltage')
        elif sensor_type == 'vibration':
            # For vibration, check for any magnitude column as a fallback
            magnitude_cols = [col for col in df.columns if 'Magnitude' in col and 'Threshold' not in col]
            if magnitude_cols:
                required_columns.append(magnitude_cols[0])
            else:
                required_columns.append('MagnitudeRMSInMilliG')  # Default
                
        missing_columns = [col for col in required_columns if col not in df.columns]
            
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
                
        # Check data types
        if 'EventTimeUtc' in df.columns:
            try:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df['EventTimeUtc']):
                    df['EventTimeUtc'] = pd.to_datetime(df['EventTimeUtc'], errors='coerce')
                    # Check if any rows have valid timestamps
                    if df['EventTimeUtc'].isna().all():
                        return False, "All timestamps are invalid"
            except:
                return False, "Could not convert EventTimeUtc to datetime"
                    
        # Check for all NaN values in key columns
        for col in required_columns:
            if col != 'EventTimeUtc' and col in df.columns and df[col].isna().all():
                return False, f"Column {col} contains all NaN values"
                    
        # Check for sufficient data
        if len(df) < 5:
            return False, "Insufficient data points (minimum 5 required)"
                
        return True, "Data validation passed"
            
    def visualize_data(self, df, sensor_type, sensor_id=None):
        """Generate visualization for sensor data"""
        if df is None or df.empty:
            return None
        
        try:
            # Filter by sensor_id if provided
            if sensor_id:
                df = df[df['SensorId'] == sensor_id]
                
            if df.empty:
                return None
                
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Determine value column based on sensor type
            value_col = None
            title = f'{sensor_type.capitalize()} Data'
            
            if sensor_type == 'temperature':
                value_col = 'TemperatureInCelsius'
                title = 'Temperature Data'
            elif sensor_type == 'battery':
                value_col = 'BatteryVoltage'
                title = 'Battery Voltage Data'
            elif sensor_type == 'vibration':
                # Find any magnitude column
                magnitude_cols = [col for col in df.columns if 'Magnitude' in col and 'Threshold' not in col]
                if magnitude_cols:
                    value_col = magnitude_cols[0]
                    title = 'Vibration Magnitude Data'
            elif sensor_type == 'motion':
                if 'daily_event_count' in df.columns:
                    value_col = 'daily_event_count'
                    title = 'Motion Event Count'
                else:
                    # Look for any magnitude column
                    magnitude_cols = [col for col in df.columns if 'Magnitude' in col or 'magnitude' in col.lower()]
                    if magnitude_cols:
                        value_col = magnitude_cols[0]
                        title = 'Motion Magnitude Data'
            
            # If no suitable value column found, can't create visualization
            if not value_col or value_col not in df.columns:
                self.logger.warning(f"No suitable value column found for {sensor_type} visualization")
                plt.close()
                return None
            
            # Ensure event_time is datetime
            time_col = 'EventTimeUtc'
            if time_col in df.columns:
                try:
                    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                except Exception as e:
                    self.logger.warning(f"Error converting timestamps for visualization: {str(e)}")
                    plt.close()
                    return None
            else:
                self.logger.warning(f"No timestamp column found for visualization")
                plt.close()
                return None
            
            # Plot data by sensor
            if sensor_id is None and len(df['SensorId'].unique()) > 1:
                # Multiple sensors - plot each one
                for sid in df['SensorId'].unique()[:5]:  # Limit to 5 sensors for readability
                    sensor_data = df[df['SensorId'] == sid]
                    plt.plot(sensor_data[time_col], sensor_data[value_col], label=f'Sensor {sid}')
                plt.legend()
            else:
                # Single sensor or specific sensor selected
                plt.plot(df[time_col], df[value_col])
            
            # Add labels and title
            plt.xlabel('Time')
            plt.ylabel(value_col)
            plt.title(title)
            plt.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            buf.seek(0)
            
            # Convert to base64 for web display
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            plt.close()
            return img_str
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
            if 'plt' in locals():
                plt.close()
            return None
            
    def optimize_memory(self, df):
        """Optimize dataframe memory usage"""
        if df is None or df.empty:
            return df
                
        # Check if optimization is enabled
        if not self.config['memory_optimization'].get('optimize_dtypes', True):
            return df
                
        try:
            # Create a copy to avoid warnings
            result = df.copy()
                
            # Optimize numeric columns
            for col in result.select_dtypes(include=['int']).columns:
                # Downcast integers
                result[col] = pd.to_numeric(result[col], downcast='integer')
                    
            for col in result.select_dtypes(include=['float']).columns:
                # Downcast floats
                result[col] = pd.to_numeric(result[col], downcast='float')
                    
            # Convert object columns to categorical if appropriate
            for col in result.select_dtypes(include=['object']).columns:
                # Check if column has low cardinality (few unique values)
                n_unique = result[col].nunique()
                if n_unique > 0 and n_unique < len(result) * 0.5:  # Less than 50% unique values
                    result[col] = result[col].astype('category')
                        
            return result
                
        except Exception as e:
            self.logger.error(f"Error optimizing memory usage: {str(e)}")
            return df