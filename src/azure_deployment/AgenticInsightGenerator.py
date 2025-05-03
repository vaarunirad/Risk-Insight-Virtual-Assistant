import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
import os
import json
import requests
import traceback

try:
    # For OpenAI v1.0.0 and higher
    from openai import AzureOpenAI
except ImportError:
    # For older versions
    import openai
    AzureOpenAI = None

class AgenticInsightGenerator:
    """Generate natural language insights from anomaly detection results.
    
    This class uses Azure OpenAI to generate human-readable insights from
    anomaly detection results. It can generate:
    1. Short-term insights for immediate anomalies
    2. Long-term insights for trends and patterns
    3. Relationship insights for connected anomalies
    """
    
    def __init__(self, config=None, logger=None):
        """Initialize the insight generator"""
        self.default_config = {
            'api': {
                'use_azure_openai': True,
                'azure_endpoint': os.environ.get('OPENAI_ENDPOINT'),
                'azure_deployment': os.environ.get('OPENAI_DEPLOYMENT', 'gpt-mini-04'),
                'temperature': 0.2,
                'max_tokens': 500,
                'top_p': 0.95,
                'frequency_penalty': 0.0,
                'presence_penalty': 0.0,
                'timeout': 30
            },
            'threshold_priority': {
                'always_critical': True,
                'boost_factor': 0.5
            },
            'motion_analysis': {
                'treat_all_as_anomalies': True,
                'motion_event_boost': 0.4
            },
            'asset_mapping': {
                'use_asset_info': True,
                'default_description': 'sensor'
            },
            'memory_optimization': {
                'batch_size': 20,
                'max_context_size': 4000
            }
        }
        
        self.config = config or self.default_config
        self.logger = logger or logging.getLogger('agentic_insight_generator')
        self.client = self.setup_openai_client()
        self.asset_mapping = {}
        
    def setup_openai_client(self):
        """Set up the OpenAI client with error handling"""
        api_config = self.config.get('api', {})
        
        try:
            # Check for personal OpenAI API key first
            personal_api_key = os.environ.get('PERSONAL_OPENAI_API_KEY')
            if personal_api_key:
                self.logger.info("Using personal OpenAI API key")
                try:
                    # For OpenAI v1.0.0 and above
                    from openai import OpenAI
                    client = OpenAI(api_key=personal_api_key)
                    return client
                except ImportError:
                    # For older versions
                    import openai
                    openai.api_key = personal_api_key
                    return openai
                    
            # Check if Azure OpenAI is configured
            if api_config.get('use_azure_openai', True):
                # Check for Azure endpoint
                azure_endpoint = api_config.get('azure_endpoint')
                if not azure_endpoint:
                    self.logger.warning("Azure OpenAI endpoint not provided, checking environment variables")
                    azure_endpoint = os.environ.get('OPENAI_ENDPOINT')
                    
                if not azure_endpoint:
                    self.logger.error("Azure OpenAI endpoint not found in config or environment")
                    return None
                
                # Get API key from environment or config
                api_key = os.environ.get('OPENAI_API_KEY')
                
                if not api_key:
                    self.logger.error("API key not found in environment")
                    return None
                
                # Check if we have the Azure OpenAI client
                if AzureOpenAI:
                    # Setup client with proper error handling
                    try:
                        client = AzureOpenAI(
                            azure_endpoint=azure_endpoint,
                            api_key=api_key,
                            api_version="2024-02-01"
                        )
                        # Verify client works by making a minimal test call
                        self.logger.info("Successfully created Azure OpenAI client")
                        return client
                    except Exception as e:
                        self.logger.error(f"Error initializing Azure OpenAI client: {str(e)}")
                        return None
                else:
                    # Try to use the old client format if AzureOpenAI isn't imported
                    try:
                        if hasattr(openai, 'AzureOpenAI'):
                            client = openai.AzureOpenAI(
                                azure_endpoint=azure_endpoint,
                                api_key=api_key,
                                api_version="2024-02-01"
                            )
                        else:
                            # Fall back to standard client with Azure configuration
                            openai.api_type = "azure"
                            openai.api_base = azure_endpoint
                            openai.api_version = "2024-02-01"
                            openai.api_key = api_key
                            client = openai
                            
                        self.logger.info("Successfully created OpenAI client with Azure config")
                        return client
                    except Exception as e:
                        self.logger.error(f"Error initializing OpenAI client with Azure config: {str(e)}")
                        return None
            else:
                self.logger.warning("Azure OpenAI not configured, insights generation will be limited")
                return None
        except Exception as e:
            self.logger.error(f"Unexpected error setting up OpenAI client: {str(e)}")
            return None
                
    def create_asset_mapping(self, sensors_df, assets_df):
        """Create mapping between sensors and assets"""
        mapping = {}
        
        # Check if dataframes are valid
        if sensors_df is None or sensors_df.empty or assets_df is None or assets_df.empty:
            self.logger.warning("Cannot create asset mapping: missing or empty dataframes")
            return mapping
        
        try:
            # Determine sensor ID column
            sensor_id_col = None
            if 'sensor_id' in sensors_df.columns:
                sensor_id_col = 'sensor_id'
            elif 'BluetoothAddress' in sensors_df.columns:
                sensor_id_col = 'BluetoothAddress'
            elif 'SensorId' in sensors_df.columns:
                sensor_id_col = 'SensorId'
            
            if not sensor_id_col:
                self.logger.warning("No sensor ID column found in sensors_df")
                return mapping
            
            # Check if asset ID columns exist
            if "AssetId" in sensors_df.columns and "AssetId" in assets_df.columns:
                self.logger.info("Creating asset mapping using AssetId")
                
                # Join dataframes on AssetId
                try:
                    sensor_assets = pd.merge(
                        sensors_df, 
                        assets_df[["AssetId", "Description"]], 
                        on="AssetId", 
                        how="left"
                    )
                    
                    # Create mapping
                    for _, row in sensor_assets.iterrows():
                        sensor_id = row.get(sensor_id_col)
                        description = row.get("Description")
                        asset_id = row.get("AssetId")
                        
                        if pd.notna(sensor_id) and pd.notna(asset_id):
                            mapping[sensor_id] = {
                                "asset_id": asset_id,
                                "description": description if pd.notna(description) else "Unknown asset"
                            }
                except Exception as e:
                    self.logger.warning(f"Error creating asset mapping: {str(e)}")
            else:
                self.logger.warning("Cannot create asset mapping: missing AssetId column")
                
        except Exception as e:
            self.logger.error(f"Error in create_asset_mapping: {str(e)}")
            
        self.asset_mapping = mapping
        self.logger.info(f"Created asset mapping for {len(mapping)} sensors")
        
        return mapping
        
    def generate_all_insights(self, dfs, relationship_data, sensors_df, assets_df=None):
        """Generate all types of insights"""
        if not dfs:
            self.logger.warning("No data provided for insight generation")
            return {
                'short_term': [],
                'long_term': [],
                'relationships': []
            }
            
        # Create asset mapping if assets_df is provided
        try:
            if assets_df is not None and not assets_df.empty:
                self.create_asset_mapping(sensors_df, assets_df)
        except Exception as e:
            self.logger.warning(f"Error creating asset mapping: {str(e)}")
        
        # Generate insights with error handling for each type
        short_term_insights = []
        long_term_insights = []
        relationship_insights = []
        
        try:
            short_term_insights = self.generate_short_term_insights(dfs, sensors_df)
            self.logger.info(f"Generated {len(short_term_insights)} short-term insights")
        except Exception as e:
            self.logger.error(f"Error generating short-term insights: {str(e)}")
            traceback.print_exc()
        
        try:
            long_term_insights = self.generate_long_term_insights(dfs, sensors_df)
            self.logger.info(f"Generated {len(long_term_insights)} long-term insights")
        except Exception as e:
            self.logger.error(f"Error generating long-term insights: {str(e)}")
            traceback.print_exc()
        
        try:
            relationship_insights = relationship_data if relationship_data else []
            self.logger.info(f"Using {len(relationship_insights)} relationship insights")
        except Exception as e:
            self.logger.error(f"Error processing relationship insights: {str(e)}")
            traceback.print_exc()
        
        # Combine all insights
        all_insights = {
            'short_term': short_term_insights,
            'long_term': long_term_insights,
            'relationships': relationship_insights
        }
        
        return all_insights
        
    def generate_short_term_insights(self, dfs, sensors_df):
        """Generate short-term insights for immediate anomalies"""
        insights = []
        
        # Process each sensor type
        for sensor_type, df in dfs.items():
            if df is None or df.empty:
                continue
                
            # Skip if sensor_id column is missing
            if 'sensor_id' not in df.columns:
                self.logger.warning(f"Missing 'sensor_id' column in {sensor_type} data, skipping")
                continue
            
            # Get anomaly columns
            anomaly_cols = self.get_anomaly_columns(df)
            if not anomaly_cols:
                self.logger.warning(f"No anomaly columns found in {sensor_type} data")
                continue
                
            # Create a mask for any anomaly
            any_anomaly = df[anomaly_cols].any(axis=1)
            
            # Skip if no anomalies
            if not any_anomaly.any():
                self.logger.info(f"No anomalies detected in {sensor_type} data")
                continue
            
            # Get sensors with anomalies
            anomaly_sensors = df.loc[any_anomaly, 'sensor_id'].unique()
            
            # Process sensors in batches
            batch_size = self.config.get('memory_optimization', {}).get('batch_size', 20)
            
            for i in range(0, len(anomaly_sensors), batch_size):
                batch_sensors = anomaly_sensors[i:i+batch_size]
                
                # Process each sensor in this batch
                for sensor_id in batch_sensors:
                    try:
                        # Get sensor data
                        sensor_df = df[df['sensor_id'] == sensor_id]
                        
                        # Skip if no anomalies for this sensor
                        sensor_anomalies = sensor_df[anomaly_cols].any(axis=1)
                        if not sensor_anomalies.any():
                            continue
                            
                        # Get sensor info
                        sensor_name = sensor_id
                        device_id = sensor_id
                        account_id = None
                        
                        # Try to get info from sensors_df
                        if sensors_df is not None and not sensors_df.empty:
                            # Determine the sensor ID column in sensors_df
                            sensor_id_col = None
                            if 'sensor_id' in sensors_df.columns:
                                sensor_id_col = 'sensor_id'
                            elif 'BluetoothAddress' in sensors_df.columns:
                                sensor_id_col = 'BluetoothAddress'
                            elif 'SensorId' in sensors_df.columns:
                                sensor_id_col = 'SensorId'
                            
                            if sensor_id_col:
                                sensor_info = sensors_df[sensors_df[sensor_id_col] == sensor_id]
                                
                                if not sensor_info.empty:
                                    # Get device ID if available
                                    device_id_col = None
                                    if 'DeviceId' in sensor_info.columns:
                                        device_id_col = 'DeviceId'
                                    elif 'device_id' in sensor_info.columns:
                                        device_id_col = 'device_id'
                                    
                                    if device_id_col and not sensor_info[device_id_col].isna().all():
                                        device_id = sensor_info[device_id_col].iloc[0]
                                    
                                    # Get account ID if available
                                    account_id_col = None
                                    if 'account_id' in sensor_info.columns:
                                        account_id_col = 'account_id'
                                    elif 'AccountId' in sensor_info.columns:
                                        account_id_col = 'AccountId'
                                    
                                    if account_id_col and not sensor_info[account_id_col].isna().all():
                                        account_id = sensor_info[account_id_col].iloc[0]
                                    
                                    # Get description if available
                                    description_col = None
                                    if 'Description' in sensor_info.columns:
                                        description_col = 'Description'
                                    elif 'description' in sensor_info.columns:
                                        description_col = 'description'
                                    
                                    if description_col and not sensor_info[description_col].isna().all():
                                        sensor_name = sensor_info[description_col].iloc[0]
                        
                        # Get asset info from mapping if available
                        asset_info = self.asset_mapping.get(sensor_id, {})
                        asset_description = asset_info.get('description', sensor_name)
                        
                        # Get anomaly data
                        anomaly_data = sensor_df[sensor_anomalies]
                        
                        # Get value column
                        value_col = self.get_value_column(sensor_df, sensor_type)
                        
                        if value_col and value_col in sensor_df.columns:
                            # Calculate anomaly stats
                            anomaly_values = anomaly_data[value_col].dropna()
                            normal_values = sensor_df.loc[~sensor_anomalies, value_col].dropna()
                            
                            if len(anomaly_values) > 0 and len(normal_values) > 0:
                                anomaly_mean = anomaly_values.mean()
                                normal_mean = normal_values.mean()
                                
                                if normal_mean != 0:
                                    percent_diff = ((anomaly_mean - normal_mean) / abs(normal_mean)) * 100
                                else:
                                    percent_diff = 0 if anomaly_mean == 0 else 100
                                    
                                # Check for threshold violations
                                threshold_violations = False
                                if 'threshold_violation' in anomaly_data.columns:
                                    threshold_violations = anomaly_data['threshold_violation'].any()
                                    
                                # Calculate anomaly score
                                anomaly_score = 0.0
                                
                                # Use isolation forest score if available
                                if 'isolation_forest_score' in anomaly_data.columns:
                                    forest_score = anomaly_data['isolation_forest_score'].max()
                                    if not pd.isna(forest_score):
                                        anomaly_score = max(anomaly_score, forest_score)
                                        
                                # Use time series score if available
                                if 'time_series_score' in anomaly_data.columns:
                                    ts_score = anomaly_data['time_series_score'].max()
                                    if not pd.isna(ts_score):
                                        anomaly_score = max(anomaly_score, ts_score)
                                        
                                # Use z-score if available
                                if 'z_score' in anomaly_data.columns:
                                    max_z = anomaly_data['z_score'].abs().max()
                                    if not pd.isna(max_z):
                                        # Normalize z-score to 0-1 range, cap at 10 for normalization
                                        z_score_norm = min(max_z, 10) / 10
                                        anomaly_score = max(anomaly_score, z_score_norm)
                                        
                                # Boost score for threshold violations
                                if threshold_violations and self.config.get('threshold_priority', {}).get('always_critical', True):
                                    boost = self.config.get('threshold_priority', {}).get('boost_factor', 0.5)
                                    anomaly_score = min(1.0, anomaly_score + boost)
                                
                                # Generate insight text with fallback to template
                                try:
                                    insight_text = self.generate_insight_text(
                                        sensor_type=sensor_type,
                                        sensor_name=sensor_name,
                                        device_id=device_id,
                                        asset_description=asset_description,
                                        anomaly_data=anomaly_data,
                                        value_col=value_col,
                                        anomaly_mean=anomaly_mean,
                                        normal_mean=normal_mean,
                                        percent_diff=percent_diff,
                                        threshold_violations=threshold_violations,
                                        anomaly_score=anomaly_score
                                    )
                                except Exception as e:
                                    self.logger.warning(f"Error generating insight text: {str(e)}")
                                    # Fall back to template
                                    insight_text = self.generate_template_insight(
                                        sensor_type, sensor_name, device_id, asset_description,
                                        anomaly_mean, normal_mean, percent_diff, threshold_violations,
                                        anomaly_score
                                    )
                                
                                # Count multi-detection instances
                                detection_methods = 0
                                if 'statistical_anomaly' in anomaly_data.columns and anomaly_data['statistical_anomaly'].any():
                                    detection_methods += 1
                                if 'isolation_forest_anomaly' in anomaly_data.columns and anomaly_data['isolation_forest_anomaly'].any():
                                    detection_methods += 1
                                if 'time_series_anomaly' in anomaly_data.columns and anomaly_data['time_series_anomaly'].any():
                                    detection_methods += 1
                                
                                # Determine severity
                                if threshold_violations:
                                    severity = 'critical'
                                elif anomaly_score >= 0.8:
                                    severity = 'critical'
                                elif anomaly_score >= 0.6 or (detection_methods >= 2 and anomaly_score >= 0.4):
                                    severity = 'concerning'
                                elif anomaly_score >= 0.4:
                                    severity = 'moderate' 
                                else:
                                    severity = 'minor'
                                
                                # Create insight object
                                insight = {
                                    'sensor_id': sensor_id,
                                    'device_id': device_id,
                                    'sensor_type': sensor_type,
                                    'text': insight_text,
                                    'anomaly_score': float(anomaly_score),
                                    'timestamp': datetime.now().isoformat(),
                                    'anomaly_count': int(sensor_anomalies.sum()),
                                    'total_count': len(sensor_df),
                                    'anomaly_percentage': float(sensor_anomalies.sum() / len(sensor_df) * 100),
                                    'multi_detection': detection_methods >= 2,
                                    'detection_methods': detection_methods,
                                    'severity': severity,
                                    'asset_name': asset_description
                                }
                                
                                # Add asset ID if available
                                if 'asset_id' in asset_info:
                                    insight['asset_id'] = asset_info['asset_id']
                                    
                                # Add account ID if available
                                if account_id:
                                    insight['account_id'] = account_id
                                    
                                # Add to insights list
                                insights.append(insight)
                    except Exception as e:
                        self.logger.error(f"Error processing sensor {sensor_id} for insights: {str(e)}")
        
        return insights
        
    def generate_long_term_insights(self, dfs, sensors_df):
        """Generate long-term insights for trends and patterns"""
        insights = []
        
        # Process each sensor type
        for sensor_type, df in dfs.items():
            if df is None or df.empty:
                continue
                
            # Skip if sensor_id column is missing
            if 'sensor_id' not in df.columns:
                self.logger.warning(f"Missing 'sensor_id' column in {sensor_type} data, skipping")
                continue
                
            # Skip if time column is missing
            time_col = None
            if 'event_time' in df.columns:
                time_col = 'event_time'
            elif 'EventTimeUtc' in df.columns:
                time_col = 'EventTimeUtc'
                
            if not time_col:
                self.logger.warning(f"No time column found in {sensor_type} data, skipping trend analysis")
                continue
            
            # Get value column
            value_col = self.get_value_column(df, sensor_type)
            if not value_col:
                self.logger.warning(f"No value column found in {sensor_type} data, skipping trend analysis")
                continue
                
            # Process each sensor
            for sensor_id in df['sensor_id'].unique():
                try:
                    # Get sensor data
                    sensor_df = df[df['sensor_id'] == sensor_id].copy()
                    
                    # Skip if not enough data points
                    if len(sensor_df) < 10:
                        continue
                        
                    # Skip if value column doesn't have enough data
                    if sensor_df[value_col].isna().sum() > len(sensor_df) * 0.5:
                        continue
                        
                    # Ensure data is sorted by time
                    try:
                        # Convert to datetime if not already
                        if not pd.api.types.is_datetime64_any_dtype(sensor_df[time_col]):
                            sensor_df[time_col] = pd.to_datetime(sensor_df[time_col])
                        
                        # Sort by time
                        sensor_df = sensor_df.sort_values(time_col)
                    except Exception as e:
                        self.logger.debug(f"Error sorting data for sensor {sensor_id}: {str(e)}")
                        # Skip if we can't sort by time
                        continue
                        
                    # Calculate trend (simple linear regression)
                    try:
                        # Get data points
                        y = sensor_df[value_col].values
                        x = np.arange(len(y))
                        
                        # Skip if too many NaN values
                        if np.isnan(y).sum() > 0.2 * len(y):
                            continue
                            
                        # Basic linear regression
                        # y = mx + b
                        n = len(x)
                        if n <= 1:
                            continue  # Need at least 2 points
                            
                        m = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
                        b = (np.sum(y) - m * np.sum(x)) / n
                        
                        # Calculate r-squared
                        y_pred = m * x + b
                        ss_total = np.sum((y - np.mean(y))**2)
                        ss_residual = np.sum((y - y_pred)**2)
                        r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
                        
                        # Get start and end values for trend description
                        start_val = y[0]
                        end_val = y[-1]
                        
                        # Calculate percentage change
                        if abs(start_val) > 0.001:  # Avoid division by very small numbers
                            pct_change = ((end_val - start_val) / abs(start_val)) * 100
                        else:
                            pct_change = 0
                            
                        # Calculate time range
                        if time_col in sensor_df.columns:
                            start_time = sensor_df[time_col].iloc[0]
                            end_time = sensor_df[time_col].iloc[-1]
                            days = (end_time - start_time).total_seconds() / (24 * 3600)
                        else:
                            days = 0
                            
                        # Skip if trend isn't significant
                        if abs(pct_change) < 10 and r_squared < 0.4:
                            continue
                            
                        # Calculate trend score (normalized)
                        trend_score = min(1.0, (abs(pct_change) / 100) * r_squared)
                        
                        # Get sensor info
                        sensor_name = sensor_id
                        device_id = sensor_id
                        account_id = None
                        
                        # Try to get info from sensors_df
                        if sensors_df is not None and not sensors_df.empty:
                            # Determine the sensor ID column in sensors_df
                            sensor_id_col = None
                            if 'sensor_id' in sensors_df.columns:
                                sensor_id_col = 'sensor_id'
                            elif 'BluetoothAddress' in sensors_df.columns:
                                sensor_id_col = 'BluetoothAddress'
                            elif 'SensorId' in sensors_df.columns:
                                sensor_id_col = 'SensorId'
                            
                            if sensor_id_col:
                                sensor_info = sensors_df[sensors_df[sensor_id_col] == sensor_id]
                                
                                if not sensor_info.empty:
                                    # Get device ID if available
                                    device_id_col = None
                                    if 'DeviceId' in sensor_info.columns:
                                        device_id_col = 'DeviceId'
                                    elif 'device_id' in sensor_info.columns:
                                        device_id_col = 'device_id'
                                    
                                    if device_id_col and not sensor_info[device_id_col].isna().all():
                                        device_id = sensor_info[device_id_col].iloc[0]
                                    
                                    # Get account ID if available
                                    account_id_col = None
                                    if 'account_id' in sensor_info.columns:
                                        account_id_col = 'account_id'
                                    elif 'AccountId' in sensor_info.columns:
                                        account_id_col = 'AccountId'
                                    
                                    if account_id_col and not sensor_info[account_id_col].isna().all():
                                        account_id = sensor_info[account_id_col].iloc[0]
                                    
                                    # Get description if available
                                    description_col = None
                                    if 'Description' in sensor_info.columns:
                                        description_col = 'Description'
                                    elif 'description' in sensor_info.columns:
                                        description_col = 'description'
                                    
                                    if description_col and not sensor_info[description_col].isna().all():
                                        sensor_name = sensor_info[description_col].iloc[0]
                        
                        # Get asset info from mapping if available
                        asset_info = self.asset_mapping.get(sensor_id, {})
                        asset_description = asset_info.get('description', sensor_name)
                        
                        # Generate trend insight text
                        try:
                            insight_text = self.generate_trend_insight_text(
                                sensor_type=sensor_type,
                                sensor_name=sensor_name,
                                device_id=device_id,
                                asset_description=asset_description,
                                trend_percent=pct_change,
                                trend_score=trend_score,
                                first_value=start_val,
                                last_value=end_val,
                                value_col=value_col,
                                days=days
                            )
                        except Exception as e:
                            self.logger.warning(f"Error generating trend insight text: {str(e)}")
                            # Fall back to template
                            insight_text = self.generate_template_trend_insight(
                                sensor_type, sensor_name, device_id, asset_description,
                                pct_change, start_val, end_val
                            )
                        
                        # Determine severity based on trend score
                        if trend_score > 0.7:
                            severity = 'concerning'
                        elif trend_score > 0.5:
                            severity = 'moderate'
                        else:
                            severity = 'minor'
                            
                        # Determine trend direction
                        trend_direction = 'increasing' if pct_change > 0 else 'decreasing'
                        
                        # Create insight object
                        insight = {
                            'sensor_id': sensor_id,
                            'device_id': device_id,
                            'sensor_type': sensor_type,
                            'text': insight_text,
                            'trend_score': float(trend_score),
                            'trend_direction': trend_direction,
                            'trend_percent': float(pct_change),
                            'days': float(days) if days else None,
                            'r_squared': float(r_squared),
                            'timestamp': datetime.now().isoformat(),
                            'severity': severity,
                            'asset_name': asset_description
                        }
                        
                        # Add asset ID if available
                        if 'asset_id' in asset_info:
                            insight['asset_id'] = asset_info['asset_id']
                            
                        # Add account ID if available
                        if account_id:
                            insight['account_id'] = account_id
                            
                        # Add to insights list
                        insights.append(insight)
                        
                    except Exception as e:
                        self.logger.debug(f"Error calculating trend for sensor {sensor_id}: {str(e)}")
                except Exception as e:
                    self.logger.warning(f"Error processing sensor {sensor_id} for trend insights: {str(e)}")
        
        return insights
        
    def generate_insight_text(self, sensor_type, sensor_name, device_id, asset_description, anomaly_data, 
                            value_col, anomaly_mean, normal_mean, percent_diff, 
                            threshold_violations, anomaly_score):
        """Generate insight text using OpenAI with enhanced asset context"""
        if not self.client:
            # Fallback to template-based insights if no OpenAI client
            return self.generate_template_insight(
                sensor_type, sensor_name, device_id, asset_description, anomaly_mean, 
                normal_mean, percent_diff, threshold_violations, anomaly_score
            )
            
        try:
            # Prepare context for the model
            context = {
                'sensor_type': sensor_type,
                'sensor_name': sensor_name,
                'device_id': device_id,
                'asset_description': asset_description,
                'anomaly_mean': float(anomaly_mean),
                'normal_mean': float(normal_mean),
                'percent_diff': float(percent_diff),
                'threshold_violations': threshold_violations,
                'anomaly_score': float(anomaly_score),
                'value_column': value_col,
                'anomaly_count': len(anomaly_data)
            }
            
            # Add units based on sensor type
            if sensor_type == 'temperature':
                context['units'] = 'Celsius'
            elif sensor_type == 'battery':
                context['units'] = 'Volts'
            elif sensor_type == 'vibration':
                context['units'] = 'milliG'
            
            # Add asset-specific context based on asset description
            asset_context = ""
            asset_lower = asset_description.lower() if asset_description else ""
            
            if "pump" in asset_lower:
                asset_context = """
                This is a pump system where:
                - Temperature increases may indicate bearing issues, cavitation, or improper lubrication
                - Vibration may indicate imbalance, misalignment, or cavitation
                - For oil refineries/chemical plants, seal integrity is critical for safety
                """
            elif "compressor" in asset_lower:
                asset_context = """
                This is a compressor where:
                - Temperature anomalies may indicate valve issues, cooling problems, or discharge pressure issues
                - Vibration could indicate bearing problems, unbalanced rotors, or mounting issues
                - In industrial settings, discharge pressure and temperature are critical parameters
                """
            elif "motor" in asset_lower:
                asset_context = """
                This is a motor where:
                - Temperature increases can indicate overloading, bearing issues, or cooling system failure
                - Vibration may suggest bearing failure, misalignment, or electrical problems
                - For critical process motors, immediate attention to overheating conditions is essential
                """
            elif "conveyor" in asset_lower:
                asset_context = """
                This is a conveyor system where:
                - Temperature increases in bearings or motors can indicate friction issues or overloading
                - Vibration may indicate bearing failure, belt misalignment, or structural issues
                - In mining operations, conveyor failures can cause significant production losses
                """
            elif "generator" in asset_lower:
                asset_context = """
                This is a generator where:
                - Temperature anomalies may indicate cooling system issues or overloading
                - Vibration could suggest bearing problems, misalignment, or mechanical failures
                - For backup power systems, reliability is critical for safety systems
                """
            elif "tank" in asset_lower:
                asset_context = """
                This is a storage tank where:
                - Temperature variations may indicate heating system issues or external environment effects
                - For chemical storage, temperature control is often critical for safety
                - In industrial processes, maintaining proper storage conditions affects product quality
                """
            
            # Add asset context to the prompt
            context['asset_context'] = asset_context
                
            # Create insight prompt
            prompt = self.create_enhanced_insight_prompt(context)
            
            # Get API config
            api_config = self.config.get('api', {})
            
            # Check for personal OpenAI API key
            personal_api_key = os.environ.get('PERSONAL_OPENAI_API_KEY')
            
            # Different call pattern based on client type
            if personal_api_key and hasattr(self.client, 'chat') and hasattr(self.client.chat, 'completions'):
                # Using personal OpenAI API with new client format
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",  # Use GPT-3.5 for personal API
                        messages=[
                            {"role": "system", "content": "You are an expert industrial sensor analyst. Your task is to generate clear, concise insights about sensor anomalies. Focus on what the anomaly means in practical terms, its potential impact, and safety implications. Be direct and use technical language appropriate for engineers."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=api_config.get('temperature', 0.2),
                        max_tokens=api_config.get('max_tokens', 500),
                        top_p=api_config.get('top_p', 0.95),
                        frequency_penalty=api_config.get('frequency_penalty', 0.0),
                        presence_penalty=api_config.get('presence_penalty', 0.0)
                    )
                    
                    # Extract insight text
                    insight_text = response.choices[0].message.content.strip()
                except Exception as e:
                    self.logger.warning(f"Error calling personal OpenAI API: {str(e)}")
                    # Fall back to template
                    return self.generate_template_insight(
                        sensor_type, sensor_name, device_id, asset_description, anomaly_mean, 
                        normal_mean, percent_diff, threshold_violations, anomaly_score
                    )
            elif hasattr(self.client, 'chat') and hasattr(self.client.chat, 'completions'):
                # New OpenAI client with Azure
                deployment = api_config.get('azure_deployment', 'gpt-mini-04')
                
                try:
                    response = self.client.chat.completions.create(
                        model=deployment,
                        messages=[
                            {"role": "system", "content": "You are an expert industrial sensor analyst. Your task is to generate clear, concise insights about sensor anomalies. Focus on what the anomaly means in practical terms, its potential impact, and safety implications. Be direct and use technical language appropriate for engineers."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=api_config.get('temperature', 0.2),
                        max_tokens=api_config.get('max_tokens', 500),
                        top_p=api_config.get('top_p', 0.95),
                        frequency_penalty=api_config.get('frequency_penalty', 0.0),
                        presence_penalty=api_config.get('presence_penalty', 0.0)
                    )
                    
                    # Extract insight text
                    insight_text = response.choices[0].message.content.strip()
                except Exception as e:
                    self.logger.warning(f"Error calling Azure OpenAI API: {str(e)}")
                    # Fall back to template
                    return self.generate_template_insight(
                        sensor_type, sensor_name, device_id, asset_description, anomaly_mean, 
                        normal_mean, percent_diff, threshold_violations, anomaly_score
                    )
            elif hasattr(self.client, 'create'):
                # Older OpenAI client
                deployment = api_config.get('azure_deployment', 'gpt-mini-04')
                
                try:
                    response = self.client.create(
                        deployment_id=deployment,
                        messages=[
                            {"role": "system", "content": "You are an expert industrial sensor analyst. Your task is to generate clear, concise insights about sensor anomalies. Focus on what the anomaly means in practical terms, its potential impact, and safety implications. Be direct and use technical language appropriate for engineers."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=api_config.get('temperature', 0.2),
                        max_tokens=api_config.get('max_tokens', 500),
                        top_p=api_config.get('top_p', 0.95),
                        frequency_penalty=api_config.get('frequency_penalty', 0.0),
                        presence_penalty=api_config.get('presence_penalty', 0.0)
                    )
                    
                    # Extract insight text
                    insight_text = response['choices'][0]['message']['content'].strip()
                except Exception as e:
                    self.logger.warning(f"Error calling OpenAI API: {str(e)}")
                    # Fall back to template
                    return self.generate_template_insight(
                        sensor_type, sensor_name, device_id, asset_description, anomaly_mean, 
                        normal_mean, percent_diff, threshold_violations, anomaly_score
                    )
            else:
                self.logger.error("Unsupported OpenAI client configuration")
                # Fall back to template
                return self.generate_template_insight(
                    sensor_type, sensor_name, device_id, asset_description, anomaly_mean, 
                    normal_mean, percent_diff, threshold_violations, anomaly_score
                )
                
            # Add prefix for threshold violations
            if threshold_violations and self.config.get('threshold_priority', {}).get('always_critical', True):
                if not insight_text.startswith('CRITICAL'):
                    insight_text = f"CRITICAL: {insight_text}"
                    
            return insight_text
            
        except Exception as e:
            self.logger.error(f"Error generating insight text: {str(e)}")
            
            # Fallback to template-based insights
            return self.generate_template_insight(
                sensor_type, sensor_name, device_id, asset_description, anomaly_mean, 
                normal_mean, percent_diff, threshold_violations, anomaly_score
            )
    
    def create_enhanced_insight_prompt(self, context):
        """Create an enhanced prompt for generating insights with asset-specific context"""
        # Get the base prompt from standard method
        base_prompt = self.create_insight_prompt(context)
        
        # Add asset-specific context if available
        asset_context = context.get('asset_context', '')
        if asset_context:
            base_prompt += f"""

            ASSET CONTEXT:
            {asset_context}
            
            Based on this asset context and sensor data, provide a concise insight explaining the anomaly, potential causes, and implications.
            """
        
        return base_prompt
            
    def create_insight_prompt(self, context):
        """Create a prompt for generating insights"""
        sensor_type = context.get('sensor_type', 'unknown')
        device_id = context.get('device_id', 'unknown')
        
        if sensor_type == 'temperature':
            prompt = f"""
            Generate a concise insight about a temperature anomaly with the following details:
            - Device ID: {device_id}
            - Sensor: {context.get('sensor_name', 'unknown')}
            - Asset: {context.get('asset_description', 'unknown')}
            - Normal temperature: {context.get('normal_mean', 0):.2f} {context.get('units', 'Celsius')}
            - Anomaly temperature: {context.get('anomaly_mean', 0):.2f} {context.get('units', 'Celsius')}
            - Percent difference: {context.get('percent_diff', 0):.2f}%
            - Threshold violation: {'Yes' if context.get('threshold_violations', False) else 'No'}
            - Anomaly score: {context.get('anomaly_score', 0):.2f} (0-1 scale, higher is more severe)
            - Number of anomalous readings: {context.get('anomaly_count', 0)}
                
            The insight should be 1-2 sentences focused on what this anomaly means and potential impact.
            """
        elif sensor_type == 'battery':
            prompt = f"""
            Generate a concise insight about a battery voltage anomaly with the following details:
            - Device ID: {device_id}
            - Sensor: {context.get('sensor_name', 'unknown')}
            - Asset: {context.get('asset_description', 'unknown')}
            - Normal voltage: {context.get('normal_mean', 0):.2f} {context.get('units', 'Volts')}
            - Anomaly voltage: {context.get('anomaly_mean', 0):.2f} {context.get('units', 'Volts')}
            - Percent difference: {context.get('percent_diff', 0):.2f}%
            - Threshold violation: {'Yes' if context.get('threshold_violations', False) else 'No'}
            - Anomaly score: {context.get('anomaly_score', 0):.2f} (0-1 scale, higher is more severe)
            - Number of anomalous readings: {context.get('anomaly_count', 0)}
            
            The insight should be 1-2 sentences focused on what this anomaly means and potential impact.
            """
        elif sensor_type == 'vibration':
            prompt = f"""
            Generate a concise insight about a vibration anomaly with the following details:
            - Device ID: {device_id}
            - Sensor: {context.get('sensor_name', 'unknown')}
            - Asset: {context.get('asset_description', 'unknown')}
            - Normal vibration: {context.get('normal_mean', 0):.2f} {context.get('units', 'milliG')}
            - Anomaly vibration: {context.get('anomaly_mean', 0):.2f} {context.get('units', 'milliG')}
            - Percent difference: {context.get('percent_diff', 0):.2f}%
            - Threshold violation: {'Yes' if context.get('threshold_violations', False) else 'No'}
            - Anomaly score: {context.get('anomaly_score', 0):.2f} (0-1 scale, higher is more severe)
            - Number of anomalous readings: {context.get('anomaly_count', 0)}
            
            The insight should be 1-2 sentences focused on what this anomaly means and potential impact.
            """
        elif sensor_type == 'motion':
            prompt = f"""
            Generate a concise insight about a motion anomaly with the following details:
            - Device ID: {device_id}
            - Sensor: {context.get('sensor_name', 'unknown')}
            - Asset: {context.get('asset_description', 'unknown')}
            - Normal motion: {context.get('normal_mean', 0):.2f}
            - Anomaly motion: {context.get('anomaly_mean', 0):.2f}
            - Percent difference: {context.get('percent_diff', 0):.2f}%
            - Anomaly score: {context.get('anomaly_score', 0):.2f} (0-1 scale, higher is more severe)
            - Number of anomalous readings: {context.get('anomaly_count', 0)}
            
            The insight should be 1-2 sentences focused on what this anomaly means and potential impact.
            """
        else:
            prompt = f"""
            Generate a concise insight about a sensor anomaly with the following details:
            - Device ID: {device_id}
            - Sensor type: {sensor_type}
            - Sensor: {context.get('sensor_name', 'unknown')}
            - Asset: {context.get('asset_description', 'unknown')}
            - Normal value: {context.get('normal_mean', 0):.2f}
            - Anomaly value: {context.get('anomaly_mean', 0):.2f}
            - Percent difference: {context.get('percent_diff', 0):.2f}%
            - Threshold violation: {'Yes' if context.get('threshold_violations', False) else 'No'}
            - Anomaly score: {context.get('anomaly_score', 0):.2f} (0-1 scale, higher is more severe)
            - Number of anomalous readings: {context.get('anomaly_count', 0)}
            
            The insight should be 1-2 sentences focused on what this anomaly means and potential impact.
            """
            
        return prompt
        
    def generate_trend_insight_text(self, sensor_type, sensor_name, device_id, asset_description,
                                trend_percent, trend_score, first_value, last_value, value_col, days=None):
        """Generate trend insight text using OpenAI with improved context"""
        if not self.client:
            # Fallback to template-based insights if no OpenAI client
            return self.generate_template_trend_insight(
                sensor_type, sensor_name, device_id, asset_description, trend_percent, first_value, last_value
            )
            
        try:
            # Prepare context for the model
            context = {
                'sensor_type': sensor_type,
                'sensor_name': sensor_name,
                'device_id': device_id,
                'asset_description': asset_description,
                'trend_percent': float(trend_percent),
                'trend_score': float(trend_score),
                'first_value': float(first_value),
                'last_value': float(last_value),
                'value_column': value_col,
                'days': days or 0
            }
            
            # Add units based on sensor type
            if sensor_type == 'temperature':
                context['units'] = 'Celsius'
            elif sensor_type == 'battery':
                context['units'] = 'Volts'
            elif sensor_type == 'vibration':
                context['units'] = 'milliG'

            # Add asset-specific context based on asset description
            asset_context = ""
            asset_lower = asset_description.lower() if asset_description else ""
            
            if "pump" in asset_lower:
                asset_context = """
                For pumps:
                - Increasing temperature trends often indicate progressive bearing failure or seal issues
                - Increasing vibration trends typically suggest bearing wear, imbalance, or cavitation
                - In critical process pumps, trend analysis helps prevent unexpected failures
                """
            elif "compressor" in asset_lower:
                asset_context = """
                For compressors:
                - Temperature trends can indicate valve efficiency decline or cooling system issues
                - Vibration trend increases often precede mechanical failures
                - For industrial processes, compressor performance directly impacts production efficiency
                """
            elif "motor" in asset_lower:
                asset_context = """
                For motors:
                - Gradual temperature increases may indicate insulation breakdown or cooling issues
                - Vibration trend changes can predict bearing failures before catastrophic damage
                - In manufacturing settings, motor reliability is critical to maintain production
                """
            elif "conveyor" in asset_lower:
                asset_context = """
                For conveyors:
                - Increasing temperature trends in bearings suggest lubrication issues or excessive load
                - Vibration trends help identify developing belt misalignment or structural issues
                - In mining operations, conveyor failures can cause extensive production losses
                """
            elif "generator" in asset_lower:
                asset_context = """
                For generators:
                - Temperature trends can indicate cooling efficiency decline or load issues
                - Vibration patterns developing over time may indicate bearing wear or alignment problems
                - For power generation, preventive maintenance based on trends is essential
                """
            elif "tank" in asset_lower:
                asset_context = """
                For storage tanks:
                - Temperature trends may indicate heating system issues or environmental changes
                - Pressure trends can help identify potential integrity issues
                - For chemical storage, maintaining stable conditions is critical for safety
                """
            
            # Add asset context to prompt creation
            context['asset_context'] = asset_context
                
            # Create prompt
            prompt = self.create_enhanced_trend_prompt(context)
            
            # Get API config
            api_config = self.config.get('api', {})
            
            # Check for personal OpenAI API key
            personal_api_key = os.environ.get('PERSONAL_OPENAI_API_KEY')
            
            # Different call pattern based on client type
            if personal_api_key and hasattr(self.client, 'chat') and hasattr(self.client.chat, 'completions'):
                # Using personal OpenAI API with new client format
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",  # Use GPT-3.5 for personal API
                        messages=[
                            {"role": "system", "content": "You are an expert industrial sensor analyst. Your task is to generate clear, concise insights about sensor trends. Focus on what the trend means in practical terms, its potential impact, and maintenance implications. Be direct and use technical language appropriate for engineers."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=api_config.get('temperature', 0.2),
                        max_tokens=api_config.get('max_tokens', 500),
                        top_p=api_config.get('top_p', 0.95),
                        frequency_penalty=api_config.get('frequency_penalty', 0.0),
                        presence_penalty=api_config.get('presence_penalty', 0.0)
                    )
                    
                    # Extract insight text
                    insight_text = response.choices[0].message.content.strip()
                except Exception as e:
                    self.logger.warning(f"Error calling personal OpenAI API for trend insight: {str(e)}")
                    # Fall back to template
                    return self.generate_template_trend_insight(
                        sensor_type, sensor_name, device_id, asset_description,
                        trend_percent, first_value, last_value
                    )
            elif hasattr(self.client, 'chat') and hasattr(self.client.chat, 'completions'):
                # New OpenAI client with Azure
                deployment = api_config.get('azure_deployment', 'gpt-mini-04')
                
                try:
                    response = self.client.chat.completions.create(
                        model=deployment,
                        messages=[
                            {"role": "system", "content": "You are an expert industrial sensor analyst. Your task is to generate clear, concise insights about sensor trends. Focus on what the trend means in practical terms, its potential impact, and maintenance implications. Be direct and use technical language appropriate for engineers."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=api_config.get('temperature', 0.2),
                        max_tokens=api_config.get('max_tokens', 500),
                        top_p=api_config.get('top_p', 0.95),
                        frequency_penalty=api_config.get('frequency_penalty', 0.0),
                        presence_penalty=api_config.get('presence_penalty', 0.0)
                    )
                    
                    # Extract insight text
                    insight_text = response.choices[0].message.content.strip()
                except Exception as e:
                    self.logger.warning(f"Error calling Azure OpenAI API for trend insight: {str(e)}")
                    # Fall back to template
                    return self.generate_template_trend_insight(
                        sensor_type, sensor_name, device_id, asset_description,
                        trend_percent, first_value, last_value
                    )
            elif hasattr(self.client, 'create'):
                # Older OpenAI client
                deployment = api_config.get('azure_deployment', 'gpt-mini-04')
                
                try:
                    response = self.client.create(
                        deployment_id=deployment,
                        messages=[
                            {"role": "system", "content": "You are an expert industrial sensor analyst. Your task is to generate clear, concise insights about sensor trends. Focus on what the trend means in practical terms, its potential impact, and maintenance implications. Be direct and use technical language appropriate for engineers."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=api_config.get('temperature', 0.2),
                        max_tokens=api_config.get('max_tokens', 500),
                        top_p=api_config.get('top_p', 0.95),
                        frequency_penalty=api_config.get('frequency_penalty', 0.0),
                        presence_penalty=api_config.get('presence_penalty', 0.0)
                    )
                    
                    # Extract insight text
                    insight_text = response['choices'][0]['message']['content'].strip()
                except Exception as e:
                    self.logger.warning(f"Error calling OpenAI API for trend insight: {str(e)}")
                    # Fall back to template
                    return self.generate_template_trend_insight(
                        sensor_type, sensor_name, device_id, asset_description,
                        trend_percent, first_value, last_value
                    )
            else:
                self.logger.error("Unsupported OpenAI client configuration for trend insights")
                # Fall back to template
                return self.generate_template_trend_insight(
                    sensor_type, sensor_name, device_id, asset_description,
                    trend_percent, first_value, last_value
                )
                
            return insight_text
                
        except Exception as e:
            self.logger.error(f"Error generating trend insight text: {str(e)}")
            
            # Fallback to template-based insights
            return self.generate_template_trend_insight(
                sensor_type, sensor_name, device_id, asset_description, trend_percent, first_value, last_value
            )
    
    def create_enhanced_trend_prompt(self, context):
        """Create an enhanced prompt for generating trend insights with asset-specific context"""
        # Get the base prompt from standard method
        base_prompt = self.create_trend_prompt(context)
        
        # Add asset-specific context if available
        asset_context = context.get('asset_context', '')
        if asset_context:
            base_prompt += f"""

            ASSET CONTEXT:
            {asset_context}
            
            Based on this asset context and the trend data, provide a concise insight explaining what this trend likely means for this specific equipment, potential causes, and maintenance implications over the next {context.get('days', 0):.1f} days if the trend continues.
            """
        
        return base_prompt
            
    def create_trend_prompt(self, context):
        """Create a prompt for generating trend insights"""
        sensor_type = context.get('sensor_type', 'unknown')
        trend_direction = "increasing" if context.get('trend_percent', 0) > 0 else "decreasing"
        device_id = context.get('device_id', 'unknown')
        
        if sensor_type == 'temperature':
            prompt = f"""
            Generate a concise insight about a temperature trend with the following details:
            - Device ID: {device_id}
            - Sensor: {context.get('sensor_name', 'unknown')}
            - Asset: {context.get('asset_description', 'unknown')}
            - Initial temperature: {context.get('first_value', 0):.2f} {context.get('units', 'Celsius')}
            - Current temperature: {context.get('last_value', 0):.2f} {context.get('units', 'Celsius')}
            - Percent change: {abs(context.get('trend_percent', 0)):.2f}% {trend_direction}
            - Trend score: {context.get('trend_score', 0):.2f} (0-1 scale, higher is more significant)
            - Time period: {context.get('days', 0):.1f} days
            
            The insight should be 1-2 sentences focused on what this trend means and potential impact.
            """
        elif sensor_type == 'battery':
            prompt = f"""
            Generate a concise insight about a battery voltage trend with the following details:
            - Device ID: {device_id}
            - Sensor: {context.get('sensor_name', 'unknown')}
            - Asset: {context.get('asset_description', 'unknown')}
            - Initial voltage: {context.get('first_value', 0):.2f} {context.get('units', 'Volts')}
            - Current voltage: {context.get('last_value', 0):.2f} {context.get('units', 'Volts')}
            - Percent change: {abs(context.get('trend_percent', 0)):.2f}% {trend_direction}
            - Trend score: {context.get('trend_score', 0):.2f} (0-1 scale, higher is more significant)
            - Time period: {context.get('days', 0):.1f} days
            
            The insight should be 1-2 sentences focused on what this trend means and potential impact.
            """
        elif sensor_type == 'vibration':
            prompt = f"""
            Generate a concise insight about a vibration trend with the following details:
            - Device ID: {device_id}
            - Sensor: {context.get('sensor_name', 'unknown')}
            - Asset: {context.get('asset_description', 'unknown')}
            - Initial vibration: {context.get('first_value', 0):.2f} {context.get('units', 'milliG')}
            - Current vibration: {context.get('last_value', 0):.2f} {context.get('units', 'milliG')}
            - Percent change: {abs(context.get('trend_percent', 0)):.2f}% {trend_direction}
            - Trend score: {context.get('trend_score', 0):.2f} (0-1 scale, higher is more significant)
            - Time period: {context.get('days', 0):.1f} days
            
            The insight should be 1-2 sentences focused on what this trend means and potential impact.
            """
        elif sensor_type == 'motion':
            prompt = f"""
            Generate a concise insight about a motion trend with the following details:
            - Device ID: {device_id}
            - Sensor: {context.get('sensor_name', 'unknown')}
            - Asset: {context.get('asset_description', 'unknown')}
            - Initial motion: {context.get('first_value', 0):.2f}
            - Current motion: {context.get('last_value', 0):.2f}
            - Percent change: {abs(context.get('trend_percent', 0)):.2f}% {trend_direction}
            - Trend score: {context.get('trend_score', 0):.2f} (0-1 scale, higher is more significant)
            - Time period: {context.get('days', 0):.1f} days
            
            The insight should be 1-2 sentences focused on what this trend means and potential impact.
            """
        else:
            prompt = f"""
            Generate a concise insight about a sensor trend with the following details:
            - Device ID: {device_id}
            - Sensor type: {sensor_type}
            - Sensor: {context.get('sensor_name', 'unknown')}
            - Asset: {context.get('asset_description', 'unknown')}
            - Initial value: {context.get('first_value', 0):.2f}
            - Current value: {context.get('last_value', 0):.2f}
            - Percent change: {abs(context.get('trend_percent', 0)):.2f}% {trend_direction}
            - Trend score: {context.get('trend_score', 0):.2f} (0-1 scale, higher is more significant)
            - Time period: {context.get('days', 0):.1f} days
            
            The insight should be 1-2 sentences focused on what this trend means and potential impact.
            """
            
        return prompt
        
    def generate_template_insight(self, sensor_type, sensor_name, device_id, asset_description, 
                                anomaly_mean, normal_mean, percent_diff, 
                                threshold_violations, anomaly_score):
        """Generate a template-based insight when OpenAI is not available, with enhanced asset specificity"""
        # Determine direction
        direction = "higher" if anomaly_mean > normal_mean else "lower"
        
        # Format values based on sensor type
        if sensor_type == 'temperature':
            units = "°C"
        elif sensor_type == 'battery':
            units = "V"
        elif sensor_type == 'vibration':
            units = "milliG"
        else:
            units = ""
            
        # Create basic insight
        insight = f"Device {device_id} ({sensor_name}) on {asset_description} shows {direction} than normal {sensor_type} readings "
        insight += f"({anomaly_mean:.2f}{units} vs normal {normal_mean:.2f}{units}, {abs(percent_diff):.1f}% difference). "
        
        # Add asset-specific context if possible
        asset_lower = asset_description.lower() if asset_description else ""
        
        additional_context = ""
        if "pump" in asset_lower and sensor_type == 'temperature':
            additional_context = "Possible causes include bearing issues, cavitation, or inadequate lubrication. "
        elif "pump" in asset_lower and sensor_type == 'vibration':
            additional_context = "May indicate imbalance, misalignment, or beginning of mechanical failure. "
        elif "compressor" in asset_lower and sensor_type == 'temperature':
            additional_context = "Could indicate valve issues, cooling problems, or discharge pressure anomalies. "
        elif "compressor" in asset_lower and sensor_type == 'vibration':
            additional_context = "May suggest bearing problems, unbalanced rotors, or mounting issues. "
        elif "motor" in asset_lower and sensor_type == 'temperature':
            additional_context = "Possible causes include overloading, bearing issues, or cooling system problems. "
        elif "conveyor" in asset_lower and sensor_type == 'vibration':
            additional_context = "Could indicate bearing failure, belt misalignment, or structural issues. "
        
        # Add severity assessment
        severity_text = ""
        if threshold_violations:
            severity_text = f"CRITICAL: {insight}{additional_context}Threshold violation detected, immediate attention required."
        elif anomaly_score > 0.8:
            severity_text = f"CRITICAL: {insight}{additional_context}Severe anomaly detected, requires immediate investigation."
        elif anomaly_score > 0.6:
            severity_text = f"{insight}{additional_context}Concerning anomaly that requires prompt attention."
        elif anomaly_score > 0.4:
            severity_text = f"{insight}{additional_context}Moderate anomaly that should be monitored."
        else:
            severity_text = f"{insight}{additional_context}Minor anomaly detected."
            
        return severity_text
        
    def generate_template_trend_insight(self, sensor_type, sensor_name, device_id, asset_description,
                                    trend_percent, first_value, last_value):
        """Generate a template-based trend insight when OpenAI is not available, with enhanced asset context"""
        # Determine direction
        direction = "increasing" if trend_percent > 0 else "decreasing"
        
        # Format values based on sensor type
        if sensor_type == 'temperature':
            units = "°C"
        elif sensor_type == 'battery':
            units = "V"
        elif sensor_type == 'vibration':
            units = "milliG"
        else:
            units = ""
            
        # Create basic insight
        insight = f"Device {device_id} ({sensor_name}) on {asset_description} shows a {direction} trend in {sensor_type} "
        insight += f"from {first_value:.2f}{units} to {last_value:.2f}{units} ({abs(trend_percent):.1f}% change). "
        
        # Add asset-specific context if possible
        asset_lower = asset_description.lower() if asset_description else ""
        
        additional_context = ""
        if "pump" in asset_lower and sensor_type == 'temperature' and direction == "increasing":
            additional_context = "Consistent temperature rise may indicate developing bearing issues or insufficient cooling. "
        elif "pump" in asset_lower and sensor_type == 'vibration' and direction == "increasing":
            additional_context = "Increasing vibration trend suggests progressive mechanical degradation or mounting issues. "
        elif "compressor" in asset_lower and sensor_type == 'temperature' and direction == "increasing":
            additional_context = "Rising temperature trend may indicate valve efficiency decline or cooling system deterioration. "
        elif "motor" in asset_lower and sensor_type == 'temperature' and direction == "increasing":
            additional_context = "Gradual temperature increase suggests possible insulation breakdown or bearing wear. "
        elif "conveyor" in asset_lower and sensor_type == 'vibration' and direction == "increasing":
            additional_context = "Progressive vibration increase may indicate developing belt misalignment or bearing wear. "
        elif sensor_type == 'battery' and direction == "decreasing":
            additional_context = "Declining voltage trend may indicate battery deterioration or charging system issues. "
        
        # Add severity based on trend percentage
        severity_text = ""
        if abs(trend_percent) > 50:
            severity_text = f"{insight}{additional_context}This significant change requires immediate investigation."
        elif abs(trend_percent) > 30:
            severity_text = f"{insight}{additional_context}This substantial change should be addressed soon."
        elif abs(trend_percent) > 15:
            severity_text = f"{insight}{additional_context}This moderate change should be monitored."
        else:
            severity_text = f"{insight}{additional_context}This change is notable but not yet concerning."
            
        return severity_text
        
    def get_value_column(self, df, sensor_type):
        """Get the value column for a sensor type"""
        if sensor_type == 'temperature':
            if 'temperature_celsius' in df.columns:
                return 'temperature_celsius'
            elif 'TemperatureInCelsius' in df.columns:
                return 'TemperatureInCelsius'
            # Look for any temperature column
            temp_cols = [col for col in df.columns if 'temp' in col.lower() or 'celsius' in col.lower()]
            if temp_cols:
                return temp_cols[0]
        elif sensor_type == 'battery':
            if 'battery_voltage' in df.columns:
                return 'battery_voltage'
            elif 'BatteryVoltage' in df.columns:
                return 'BatteryVoltage'
            # Look for any voltage column
            batt_cols = [col for col in df.columns if 'batt' in col.lower() or 'volt' in col.lower()]
            if batt_cols:
                return batt_cols[0]
        elif sensor_type == 'vibration':
            if 'vibration_magnitude' in df.columns:
                return 'vibration_magnitude'
            elif 'MagnitudeRMSInMilliG' in df.columns:
                return 'MagnitudeRMSInMilliG'
            # Look for any magnitude column
            magnitude_cols = [col for col in df.columns if 'magnitude' in col.lower() or 'vib' in col.lower()]
            if magnitude_cols:
                return magnitude_cols[0]
        elif sensor_type == 'motion':
            if 'max_magnitude' in df.columns:
                return 'max_magnitude'
            elif 'daily_event_count' in df.columns:
                return 'daily_event_count'
            elif 'MaxMagnitudeRMSInMilliG' in df.columns:
                return 'MaxMagnitudeRMSInMilliG'
            # Look for any motion-related column
            motion_cols = [col for col in df.columns if 'motion' in col.lower() or 'event' in col.lower() or 'count' in col.lower() or 'magnitude' in col.lower()]
            if motion_cols:
                return motion_cols[0]
            
        # Return None if no suitable column found
        self.logger.warning(f"Could not find value column for {sensor_type}")
        return None
        
    def get_anomaly_columns(self, df):
        """Get anomaly columns based on dataframe columns"""
        anomaly_cols = []
        
        if 'statistical_anomaly' in df.columns:
            anomaly_cols.append('statistical_anomaly')
            
        if 'isolation_forest_anomaly' in df.columns:
            anomaly_cols.append('isolation_forest_anomaly')
            
        if 'time_series_anomaly' in df.columns:
            anomaly_cols.append('time_series_anomaly')
            
        if 'threshold_violation' in df.columns:
            anomaly_cols.append('threshold_violation')
            
        # If no standard anomaly columns found, look for any with 'anomaly' in the name
        if not anomaly_cols:
            anomaly_cols = [col for col in df.columns if 'anomaly' in col.lower() or 'violation' in col.lower()]
            
        return anomaly_cols
        
    def answer_maintenance_query(self, query, equipment_type=None, related_sensors=None, user_role=None, asset_name=None):
        """Generate a response to a maintenance-related query with enhanced context awareness"""
        if not self.client:
            return "OpenAI client not available for generating maintenance advice. Please check your OpenAI configuration."
                
        try:
            # Create base prompt with original functionality
            base_prompt = f"""
            Please provide a helpful, detailed response to a maintenance query about industrial equipment. 
            The response should be practical, safety-focused, and include both immediate troubleshooting 
            steps and long-term maintenance recommendations.
            
            Query: {query}
            """
                
            if equipment_type:
                base_prompt += f"\nEquipment Type: {equipment_type}\n"
                    
            if related_sensors:
                base_prompt += "\nRelated Sensor Information:\n"
                for sensor in related_sensors:
                    sensor_type = sensor.get('sensor_type', 'unknown')
                    anomalies = sensor.get('anomalies', 0)
                    base_prompt += f"- {sensor.get('sensor_id')}: {sensor_type} sensor with {anomalies} anomalies\n"
                    # Include severity and threshold information if available
                    if 'severity' in sensor:
                        base_prompt += f"  Severity: {sensor.get('severity', 'unknown')}\n"
                    if 'value' in sensor:
                        base_prompt += f"  Current Value: {sensor.get('value', 'N/A')}\n"
                    if 'threshold' in sensor:
                        base_prompt += f"  Threshold Status: {sensor.get('threshold', 'unknown')}\n"
            
            # Add enhanced context based on asset name
            if asset_name:
                base_prompt += f"\nSpecific Asset: {asset_name}\n"
                
                # Add asset-specific context based on common asset types
                asset_lower = asset_name.lower() if asset_name else ""
                
                if "pump" in asset_lower:
                    base_prompt += """
                    Asset Context: This is a pump system. Common issues include:
                    - Cavitation (indicated by vibration and noise)
                    - Seal failures (indicated by leakage)
                    - Bearing wear (indicated by temperature and vibration)
                    - Impeller damage (indicated by reduced flow and efficiency)
                    - Motor overheating (indicated by temperature anomalies)
                    """
                elif "compressor" in asset_lower:
                    base_prompt += """
                    Asset Context: This is a compressor system. Common issues include:
                    - Valve failures (indicated by temperature and efficiency anomalies)
                    - Lubrication problems (indicated by temperature and vibration)
                    - Seal issues (indicated by pressure drops)
                    - Cooling system failures (indicated by temperature spikes)
                    - Motor or turbine drive issues (indicated by vibration patterns)
                    """
                elif "conveyor" in asset_lower:
                    base_prompt += """
                    Asset Context: This is a conveyor system. Common issues include:
                    - Belt misalignment (indicated by tracking sensors)
                    - Roller bearing failures (indicated by vibration and temperature)
                    - Drive system issues (indicated by motor temperature and current)
                    - Belt wear or damage (indicated by visual inspection sensors)
                    - Material buildup (indicated by weight sensors and motor load)
                    """
                elif "generator" in asset_lower:
                    base_prompt += """
                    Asset Context: This is a generator system. Common issues include:
                    - Cooling system failures (indicated by temperature anomalies)
                    - Bearing wear (indicated by vibration patterns)
                    - Fuel system issues (for combustion generators, indicated by pressure)
                    - Excitation system problems (indicated by voltage irregularities)
                    - Insulation degradation (indicated by electrical tests)
                    """
                elif "tank" in asset_lower:
                    base_prompt += """
                    Asset Context: This is a storage tank. Common issues include:
                    - Leakage (indicated by level sensors and pressure drops)
                    - Contamination (indicated by quality sensors)
                    - Pressure control issues (indicated by pressure sensors)
                    - Temperature control issues (indicated by temperature sensors)
                    - Valve failures (indicated by flow rate anomalies)
                    """
                elif "motor" in asset_lower:
                    base_prompt += """
                    Asset Context: This is an electric motor. Common issues include:
                    - Bearing failures (indicated by vibration and temperature)
                    - Winding issues (indicated by temperature and current)
                    - Cooling problems (indicated by temperature anomalies)
                    - Alignment issues (indicated by vibration patterns)
                    - Power quality problems (indicated by current and voltage patterns)
                    """
            
            # Add role-specific context if provided
            if user_role:
                base_prompt += f"\nUser Role: {user_role}\n"
                
                role_lower = user_role.lower()
                if "technician" in role_lower:
                    base_prompt += """
                    Please include:
                    - Specific troubleshooting steps
                    - Tools and parts needed
                    - Safety precautions during maintenance
                    - Estimated repair time
                    """
                elif "manager" in role_lower:
                    base_prompt += """
                    Please include:
                    - Impact on operations and production
                    - Resource requirements
                    - Priority level and urgency
                    - Estimated downtime
                    - Cost implications
                    """
                elif "safety" in role_lower:
                    base_prompt += """
                    Please include:
                    - Safety risks and hazards
                    - Required PPE and safety measures
                    - Compliance considerations
                    - Lockout/tagout procedures
                    - Environmental impacts
                    """
                elif "engineer" in role_lower:
                    base_prompt += """
                    Please include:
                    - Root cause analysis
                    - System-level implications
                    - Long-term solutions
                    - Preventive maintenance recommendations
                    - Engineering specifications and tolerances
                    """
                elif "executive" in role_lower:
                    base_prompt += """
                    Please include:
                    - High-level summary with minimal technical jargon
                    - Business impact assessment
                    - Resource and budget implications
                    - Risk assessment in business terms
                    - Strategic recommendations
                    """
                elif "operator" in role_lower:
                    base_prompt += """
                    Please include:
                    - Normal operating parameters
                    - Operator-level troubleshooting steps
                    - When to escalate to maintenance
                    - Safe operating procedures
                    - Monitoring guidelines
                    """
            
            # Detect if query is asking about specific sensor
            sensor_query = False
            for sensor in (related_sensors or []):
                sensor_id = sensor.get('sensor_id', '')
                if sensor_id and sensor_id in query:
                    sensor_query = True
                    base_prompt += f"\nThis query specifically mentions sensor {sensor_id}. Please focus your response on this specific sensor's readings and potential issues.\n"
                    break
            
            # Get API config
            api_config = self.config.get('api', {})
                
            # Check for personal OpenAI API key
            personal_api_key = os.environ.get('PERSONAL_OPENAI_API_KEY')
                
            # Enhance system prompt with industry knowledge
            system_prompt = """You are an expert industrial maintenance advisor with deep knowledge of oil refineries, mining operations, and manufacturing plants. 

            Your expertise includes pumps, compressors, generators, conveyors, motors, tanks, and associated sensor systems. Provide practical, detailed advice for maintenance queries, emphasizing safety and efficiency.
            
            When sensors are specifically mentioned in a query, focus on interpreting their readings and explaining the implications for the equipment.
            
            Tailor your response to the user's role and the specific asset when that information is provided.
            """
                
            # Different call pattern based on client type
            if personal_api_key and hasattr(self.client, 'chat') and hasattr(self.client.chat, 'completions'):
                # Using personal OpenAI API
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Use GPT-3.5 for personal API
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": base_prompt}
                    ],
                    temperature=api_config.get('temperature', 0.2),
                    max_tokens=api_config.get('max_tokens', 800),
                    top_p=api_config.get('top_p', 0.95),
                    frequency_penalty=api_config.get('frequency_penalty', 0.0),
                    presence_penalty=api_config.get('presence_penalty', 0.0)
                )
                    
                # Extract response text
                response_text = response.choices[0].message.content.strip()
            elif hasattr(self.client, 'chat') and hasattr(self.client.chat, 'completions'):
                # New OpenAI client with Azure
                deployment = api_config.get('azure_deployment', 'gpt-mini-04')
                    
                response = self.client.chat.completions.create(
                    model=deployment,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": base_prompt}
                    ],
                    temperature=api_config.get('temperature', 0.2),
                    max_tokens=api_config.get('max_tokens', 800),
                    top_p=api_config.get('top_p', 0.95),
                    frequency_penalty=api_config.get('frequency_penalty', 0.0),
                    presence_penalty=api_config.get('presence_penalty', 0.0)
                )
                    
                # Extract response text
                response_text = response.choices[0].message.content.strip()
            elif hasattr(self.client, 'create'):
                # Older OpenAI client
                deployment = api_config.get('azure_deployment', 'gpt-mini-04')
                    
                response = self.client.create(
                    deployment_id=deployment,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": base_prompt}
                    ],
                    temperature=api_config.get('temperature', 0.2),
                    max_tokens=api_config.get('max_tokens', 800),
                    top_p=api_config.get('top_p', 0.95),
                    frequency_penalty=api_config.get('frequency_penalty', 0.0),
                    presence_penalty=api_config.get('presence_penalty', 0.0)
                )
                    
                # Extract response text
                response_text = response['choices'][0]['message']['content'].strip()
            else:
                self.logger.error("Unsupported OpenAI client configuration")
                return "Unable to generate maintenance advice due to API configuration issues."
                    
            return response_text
                    
        except Exception as e:
            self.logger.error(f"Error generating maintenance response: {str(e)}")
            return f"Error generating maintenance response. Please try again later. Technical details: {str(e)}"
    
    def _extract_sensor_from_query(self, query):
        """Helper method to extract sensor IDs from a query"""
        # Common patterns for sensor mentions
        import re
        patterns = [
            r"sensor\s+([A-Fa-f0-9]+)",  # "sensor 33F285DF2AF3"
            r"sensor\s+id\s+([A-Fa-f0-9]+)",  # "sensor id 33F285DF2AF3"
            r"sensor\s*:\s*([A-Fa-f0-9]+)",  # "sensor: 33F285DF2AF3"
            r"device\s+([A-Fa-f0-9]+)",  # "device 33F285DF2AF3"
            r"([A-Fa-f0-9]{12})"  # Just the 12-character hex ID
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                return matches[0]
        
        return None
    
    def save_insights_to_blob(self, insights):
        """Save insights to blob storage for backup"""
        if not insights:
            self.logger.warning("No insights to save to blob storage")
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
            
            # Create timestamp for unique blob name
            timestamp = time.strftime('%Y%m%d%H%M%S')
            
            # Convert insights to JSON
            insights_json = json.dumps(insights, default=str, indent=2)
            
            # Upload to blob storage
            blob_name = f"insights/{timestamp}_insights.json"
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(insights_json.encode('utf-8'), overwrite=True)
            
            self.logger.info(f"Saved insights to blob storage: {blob_name}")
            
        except Exception as e:
            self.logger.error(f"Error saving insights to blob storage: {str(e)}")
            traceback.print_exc()