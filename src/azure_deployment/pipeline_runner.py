import os
import traceback
import sys
import time
import gc  # Added for garbage collection

try:
    print("Environment variables:")
    for key, value in os.environ.items():
        # Print but mask sensitive values
        if 'KEY' in key or 'SECRET' in key or 'PASSWORD' in key:
            print(f"{key}=***MASKED***")
        else:
            print(f"{key}={value}")
    
    print("Starting pipeline_runner.py")
    import logging
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import json
    import requests
    import io

    # Import detector classes
    from StatisticalDetector import StatisticalDetector
    from IsolationForestDetector import IsolationForestDetector
    from TimeSeriesDetector import TimeSeriesDetector
    from GraphRelationshipDetector import GraphRelationshipDetector
    from AgenticInsightGenerator import AgenticInsightGenerator
    from SensorPreprocessor import SensorPreprocessor

    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('pipeline_runner')

    def optimize_dataframe_memory(df):
        """Optimize memory usage of a dataframe"""
        if df is None or df.empty:
            return df
            
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

    def runpipeline():
        """Main function to run the entire anomaly detection pipeline"""
        logger.info("Starting multi-stage anomaly detection pipeline")
        
        try:
            # Read data from blob storage
            tempdf, battdf, vibdf, motndf, sensorsdf = loaddata()
            
            # Log the columns in each dataframe for debugging
            logger.info("Column names after loading data:")
            logger.info(f"Temperature columns: {tempdf.columns.tolist() if not tempdf.empty else 'Empty dataframe'}")
            logger.info(f"Battery columns: {battdf.columns.tolist() if not battdf.empty else 'Empty dataframe'}")
            logger.info(f"Vibration columns: {vibdf.columns.tolist() if not vibdf.empty else 'Empty dataframe'}")
            logger.info(f"Motion columns: {motndf.columns.tolist() if not motndf.empty else 'Empty dataframe'}")
            logger.info(f"Sensors columns: {sensorsdf.columns.tolist() if not sensorsdf.empty else 'Empty dataframe'}")
            
            # Optimize memory usage for large dataframes
            tempdf = optimize_dataframe_memory(tempdf)
            battdf = optimize_dataframe_memory(battdf)
            vibdf = optimize_dataframe_memory(vibdf)
            motndf = optimize_dataframe_memory(motndf)
            sensorsdf = optimize_dataframe_memory(sensorsdf)
            logger.info("Optimized dataframe memory usage")
            
            # Ensure all dataframes have required columns
            tempdf = ensure_dataframe_columns(tempdf, 
                ['SensorId', 'TemperatureInCelsius', 'EventTimeUtc'], 
                ['sensor_id', 'temperature_celsius', 'event_time'])
                
            battdf = ensure_dataframe_columns(battdf,
                ['SensorId', 'BatteryVoltage', 'EventTimeUtc'],
                ['sensor_id', 'battery_voltage', 'event_time'])
                
            vibdf = ensure_dataframe_columns(vibdf,
                ['SensorId', 'MagnitudeRMSInMilliG', 'EventTimeUtc'],
                ['sensor_id', 'vibration_magnitude', 'event_time'])
                
            motndf = ensure_dataframe_columns(motndf,
                ['SensorId', 'EventTimeUtc'],
                ['sensor_id', 'event_time'])
                
            sensorsdf = ensure_dataframe_columns(sensorsdf,
                ['BluetoothAddress', 'AccountId'],
                ['sensor_id', 'account_id'])
            
            # Log the columns after renaming
            logger.info("Column names after renaming:")
            logger.info(f"Temperature columns: {tempdf.columns.tolist() if not tempdf.empty else 'Empty dataframe'}")
            logger.info(f"Battery columns: {battdf.columns.tolist() if not battdf.empty else 'Empty dataframe'}")
            logger.info(f"Vibration columns: {vibdf.columns.tolist() if not vibdf.empty else 'Empty dataframe'}")
            logger.info(f"Motion columns: {motndf.columns.tolist() if not motndf.empty else 'Empty dataframe'}")
            logger.info(f"Sensors columns: {sensorsdf.columns.tolist() if not sensorsdf.empty else 'Empty dataframe'}")
            
            # Preprocess the data with error handling
            try:
                # Create preprocessor
                preprocessor = SensorPreprocessor()
                
                # Preprocess each data type with proper error handling
                tempdf = preprocessor.preprocess_temperature(tempdf)
                logger.info(f"Temperature preprocessing completed: {len(tempdf)} records")
            except Exception as e:
                logger.error(f"Error preprocessing temperature data: {str(e)}")
                logger.info("Continuing with original temperature data")
                # Keep original data if preprocessing fails
                
            try:
                battdf = preprocessor.preprocess_battery(battdf)
                logger.info(f"Battery preprocessing completed: {len(battdf)} records")
            except Exception as e:
                logger.error(f"Error preprocessing battery data: {str(e)}")
                logger.info("Continuing with original battery data")
                # Keep original data if preprocessing fails
                
            try:
                vibdf = preprocessor.preprocess_vibration(vibdf)
                logger.info(f"Vibration preprocessing completed: {len(vibdf)} records")
            except Exception as e:
                logger.error(f"Error preprocessing vibration data: {str(e)}")
                logger.info("Continuing with original vibration data")
                # Keep original data if preprocessing fails
                
            try:
                motndf = preprocessor.preprocess_motion(motndf)
                logger.info(f"Motion preprocessing completed: {len(motndf)} records")
            except Exception as e:
                logger.error(f"Error preprocessing motion data: {str(e)}")
                logger.info("Continuing with original motion data")
                # Keep original data if preprocessing fails
            
            # Add memory cleanup after preprocessing 
            gc.collect()
            logger.info("Released memory after preprocessing")
            
            # Double check column names after preprocessing
            tempdf = ensure_dataframe_columns(tempdf, 
                ['SensorId', 'TemperatureInCelsius', 'EventTimeUtc'], 
                ['sensor_id', 'temperature_celsius', 'event_time'])
                
            battdf = ensure_dataframe_columns(battdf,
                ['SensorId', 'BatteryVoltage', 'EventTimeUtc'],
                ['sensor_id', 'battery_voltage', 'event_time'])
                
            vibdf = ensure_dataframe_columns(vibdf,
                ['SensorId', 'MagnitudeRMSInMilliG', 'EventTimeUtc'],
                ['sensor_id', 'vibration_magnitude', 'event_time'])
                
            motndf = ensure_dataframe_columns(motndf,
                ['SensorId', 'EventTimeUtc'],
                ['sensor_id', 'event_time'])
            
            # Stage 1: Statistical Anomaly Detection
            logger.info("Running statistical detection stage...")
            statistical_detector = StatisticalDetector()
            
            # Process each data type with robust error handling
            temp_with_anomalies = safe_detect(statistical_detector, tempdf, 'temperature')
            batt_with_anomalies = safe_detect(statistical_detector, battdf, 'battery')
            vib_with_anomalies = safe_detect(statistical_detector, vibdf, 'vibration')
            motn_with_anomalies = safe_detect(statistical_detector, motndf, 'motion')
            
            # Collect statistical results
            statistical_results = {
                'temperature': statistical_detector.get_results('temperature'),
                'battery': statistical_detector.get_results('battery'),
                'vibration': statistical_detector.get_results('vibration'),
                'motion': statistical_detector.get_results('motion')
            }
            
            # Add memory cleanup after statistical detection
            gc.collect() 
            logger.info("Released memory after statistical detection")
            
            # Stage 2: Isolation Forest Detection
            logger.info("Running isolation forest detection stage...")
            isolation_forest_detector = IsolationForestDetector()
            
            temp_if_anomalies = safe_detect(isolation_forest_detector, temp_with_anomalies, 'temperature')
            batt_if_anomalies = safe_detect(isolation_forest_detector, batt_with_anomalies, 'battery')
            vib_if_anomalies = safe_detect(isolation_forest_detector, vib_with_anomalies, 'vibration')
            motn_if_anomalies = safe_detect(isolation_forest_detector, motn_with_anomalies, 'motion')
            
            # Collect isolation forest results
            isolation_forest_results = {
                'temperature': isolation_forest_detector.get_results('temperature'),
                'battery': isolation_forest_detector.get_results('battery'),
                'vibration': isolation_forest_detector.get_results('vibration'),
                'motion': isolation_forest_detector.get_results('motion')
            }
            
            # Add memory cleanup after isolation forest detection
            gc.collect()
            logger.info("Released memory after isolation forest detection")
            
            # Stage 3: Time Series Detection
            logger.info("Running time series detection stage...")
            time_series_detector = TimeSeriesDetector()
            
            temp_ts_anomalies = safe_detect(time_series_detector, temp_if_anomalies, 'temperature')
            batt_ts_anomalies = safe_detect(time_series_detector, batt_if_anomalies, 'battery')
            vib_ts_anomalies = safe_detect(time_series_detector, vib_if_anomalies, 'vibration')
            motn_ts_anomalies = safe_detect(time_series_detector, motn_if_anomalies, 'motion')
            
            # Collect time series results
            time_series_results = {
                'temperature': time_series_detector.get_results('temperature'),
                'battery': time_series_detector.get_results('battery'),
                'vibration': time_series_detector.get_results('vibration'),
                'motion': time_series_detector.get_results('motion')
            }
            
            # Add memory cleanup after time series detection
            gc.collect()
            logger.info("Released memory after time series detection")
            
            # Stage 4: Graph-based Relationship Analysis
            logger.info("Running graph relationship detection stage...")

            # Create safer function to select only available columns
            def safe_select_columns(df, required_columns):
                if df is None or df.empty:
                    return df
                # Only select columns that exist in the dataframe
                available_columns = [col for col in required_columns if col in df.columns]
                if not available_columns:
                    # If none of the required columns exist, return as is
                    return df
                return df[available_columns].copy()

            try:
                # Safely select columns for each dataframe
                temp_cols = ['sensor_id', 'temperature_celsius', 'event_time', 'statistical_anomaly', 'threshold_violation']
                if 'isolation_forest_anomaly' in temp_ts_anomalies.columns:
                    temp_cols.append('isolation_forest_anomaly')
                if 'time_series_anomaly' in temp_ts_anomalies.columns:
                    temp_cols.append('time_series_anomaly')
                temp_ts_anomalies = safe_select_columns(temp_ts_anomalies, temp_cols)
                
                batt_cols = ['sensor_id', 'battery_voltage', 'event_time', 'statistical_anomaly', 'threshold_violation']
                if 'isolation_forest_anomaly' in batt_ts_anomalies.columns:
                    batt_cols.append('isolation_forest_anomaly')
                if 'time_series_anomaly' in batt_ts_anomalies.columns:
                    batt_cols.append('time_series_anomaly')
                batt_ts_anomalies = safe_select_columns(batt_ts_anomalies, batt_cols)
                
                vib_cols = ['sensor_id', 'vibration_magnitude', 'event_time', 'statistical_anomaly', 'threshold_violation']
                if 'isolation_forest_anomaly' in vib_ts_anomalies.columns:
                    vib_cols.append('isolation_forest_anomaly')
                if 'time_series_anomaly' in vib_ts_anomalies.columns:
                    vib_cols.append('time_series_anomaly')
                vib_ts_anomalies = safe_select_columns(vib_ts_anomalies, vib_cols)
                
                motn_cols = ['sensor_id', 'event_time', 'statistical_anomaly', 'threshold_violation']
                if 'time_series_anomaly' in motn_ts_anomalies.columns:
                    motn_cols.append('time_series_anomaly')
                motn_ts_anomalies = safe_select_columns(motn_ts_anomalies, motn_cols)
                
            except Exception as e:
                logger.error(f"Error selecting columns for graph detection: {str(e)}")
                # Create minimal dataframes if needed
                if temp_ts_anomalies.empty:
                    temp_ts_anomalies = pd.DataFrame(columns=['sensor_id', 'temperature_celsius', 'event_time', 'statistical_anomaly', 'threshold_violation'])
                if batt_ts_anomalies.empty:
                    batt_ts_anomalies = pd.DataFrame(columns=['sensor_id', 'battery_voltage', 'event_time', 'statistical_anomaly', 'threshold_violation'])
                if vib_ts_anomalies.empty:
                    vib_ts_anomalies = pd.DataFrame(columns=['sensor_id', 'vibration_magnitude', 'event_time', 'statistical_anomaly', 'threshold_violation'])
                if motn_ts_anomalies.empty:
                    motn_ts_anomalies = pd.DataFrame(columns=['sensor_id', 'event_time', 'statistical_anomaly', 'threshold_violation'])
                        
                    
            combined_data = {
                'temperature': temp_ts_anomalies,
                'battery': batt_ts_anomalies,
                'vibration': vib_ts_anomalies,
                'motion': motn_ts_anomalies
            }
            
            # Simplify sensors dataframe to only needed columns 
            sensorsdf_min = sensorsdf[['sensor_id', 'DeviceId', 'account_id', 'Description']].copy() if 'Description' in sensorsdf.columns else sensorsdf[['sensor_id', 'DeviceId', 'account_id']].copy()
            
            graph_detector = GraphRelationshipDetector()
            try:
                relationship_insights = graph_detector.detect(combined_data, sensorsdf_min)
                logger.info(f"Generated {len(relationship_insights)} relationship insights")
            except Exception as e:
                logger.error(f"Error in graph relationship detection: {str(e)}")
                relationship_insights = []
            
            # Store results from all stages
            final_results = {
                'statistical': statistical_results,
                'isolation_forest': isolation_forest_results,
                'time_series': time_series_results,
                'graph_relationship': {
                    'insights': relationship_insights
                }
            }
            
            # Add memory cleanup before insights generation
            gc.collect()
            logger.info("Released memory before insights generation")
            
            # Generate insights using the agentic AI
            logger.info("Generating insights...")
            try:
                insight_generator = AgenticInsightGenerator(
                    config={
                        'api': {
                            'use_azure_openai': True,
                            'azure_endpoint': os.environ.get('OPENAI_ENDPOINT'),
                            'azure_deployment': os.environ.get('OPENAI_DEPLOYMENT'),
                            'temperature': 0.2
                        },
                        'threshold_priority': {
                            'always_critical': True,  # Flag threshold violations as critical
                            'boost_factor': 0.5
                        },
                        'motion_analysis': {
                            'treat_all_as_anomalies': True,
                            'motion_event_boost': 0.4
                        }
                    }
                )
                
                insights = insight_generator.generate_all_insights(
                    dfs=combined_data,
                    relationship_data=relationship_insights,
                    sensors_df=sensorsdf_min
                )
                
                # Add severity labels based on anomaly scores and multiple detections
                insights = add_severity_labels(insights, combined_data)
                
                # Save insights to database
                save_insights(insights)
                
                # Save insights to blob as backup
                try:
                    insight_generator.save_insights_to_blob(insights)
                    logger.info("Saved insights to blob storage")
                except Exception as e:
                    logger.error(f"Error saving insights to blob: {str(e)}")
                
                # Trigger alerts for critical insights
                trigger_alerts_for_critical_insights(insights)
                
                logger.info(f"Generated and saved insights successfully")
                
            except Exception as e:
                logger.error(f"Error generating insights: {str(e)}")
                insights = None
            
            # Save processed data for future reference
            try:
                save_processed_data(combined_data)
                logger.info("Saved processed data to blob storage")
            except Exception as e:
                logger.error(f"Error saving processed data: {str(e)}")
            
            logger.info("Pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed {str(e)}")
            traceback.print_exc()
            raise

    def ensure_dataframe_columns(df, original_columns, renamed_columns):
        """Safely rename columns and ensure they exist"""
        if df is None or df.empty:
            # Create minimal empty dataframe with required columns
            empty_df = pd.DataFrame(columns=renamed_columns)
            return empty_df
        
        # Make a copy to avoid warnings
        df = df.copy()
        
        # Create mapping from original to renamed columns
        column_map = {orig: renamed for orig, renamed in zip(original_columns, renamed_columns) 
                     if orig in df.columns}
        
        # Rename columns
        df = df.rename(columns=column_map)
        
        # Check if the renamed columns exist, create them if not
        for col in renamed_columns:
            if col not in df.columns:
                logger.warning(f"Creating missing column: {col}")
                df[col] = None
        
        return df

    def safe_detect(detector, df, sensor_type):
        """Safely run detection while handling errors"""
        if df is None or df.empty or 'sensor_id' not in df.columns:
            logger.warning(f"No suitable {sensor_type} data for detection")
            # Return a minimal dataframe with required columns based on sensor type
            if sensor_type == 'temperature':
                return pd.DataFrame(columns=['sensor_id', 'temperature_celsius', 'event_time', 'statistical_anomaly', 'threshold_violation', 'z_score'])
            elif sensor_type == 'battery':
                return pd.DataFrame(columns=['sensor_id', 'battery_voltage', 'event_time', 'statistical_anomaly', 'threshold_violation', 'z_score'])
            elif sensor_type == 'vibration':
                return pd.DataFrame(columns=['sensor_id', 'vibration_magnitude', 'event_time', 'statistical_anomaly', 'threshold_violation', 'z_score'])
            elif sensor_type == 'motion':
                return pd.DataFrame(columns=['sensor_id', 'event_time', 'daily_event_count', 'statistical_anomaly', 'threshold_violation', 'z_score'])
            else:
                return pd.DataFrame(columns=['sensor_id', 'event_time', 'statistical_anomaly', 'threshold_violation', 'z_score'])
        
        try:
            result = detector.detect(df, sensor_type)
            logger.info(f"Detection completed for {sensor_type} data: {len(result)} records")
            return result
        except Exception as e:
            logger.error(f"Error in {detector.__class__.__name__} detection for {sensor_type}: {str(e)}")
            traceback.print_exc()
            # Return the original dataframe to allow the pipeline to continue
            return df

    def loaddata():
        """Load data from blob storage"""
        from azure.storage.blob import BlobServiceClient
        from azure.identity import DefaultAzureCredential
        
        print("Starting data loading process...")
        
        try:
            # Use connection string instead of managed identity
            connection_string = os.environ.get('STORAGE_CONNECTION_STRING')
            if connection_string:
                print("Using storage connection string")
                blob_service = BlobServiceClient.from_connection_string(connection_string)
            
                # Get container client
                container_name = os.environ.get('CONTAINER_NAME', 'raw-data')
                print(f"Accessing container: {container_name}")
                container_client = blob_service.get_container_client(container_name)
                
                # List available blobs for debugging
                try:
                    blobs = list(container_client.list_blobs())
                    print(f"Found {len(blobs)} blobs in container {container_name}:")
                    for blob in blobs:
                        print(f"  - {blob.name}")
                except Exception as e:
                    print(f"WARNING: Could not list blobs: {str(e)}")
                
                # Download each dataset
                print("Downloading datasets...")
                
                try:
                    tempdf = download_blob_to_dataframe(container_client, 'temperaturedata.csv')
                    print(f"Temperature data: {len(tempdf) if not tempdf.empty else 'EMPTY'} rows")
                except Exception as e:
                    print(f"Error downloading temperature data: {str(e)}")
                    tempdf = pd.DataFrame()
                
                try:
                    battdf = download_blob_to_dataframe(container_client, 'batterydata.csv')
                    print(f"Battery data: {len(battdf) if not battdf.empty else 'EMPTY'} rows")
                except Exception as e:
                    print(f"Error downloading battery data: {str(e)}")
                    battdf = pd.DataFrame()
                
                try:
                    vibdf = download_blob_to_dataframe(container_client, 'vibrationdata.csv')
                    print(f"Vibration data: {len(vibdf) if not vibdf.empty else 'EMPTY'} rows")
                except Exception as e:
                    print(f"Error downloading vibration data: {str(e)}")
                    vibdf = pd.DataFrame()
                
                try:
                    motndf = download_blob_to_dataframe(container_client, 'motiondata.csv')
                    print(f"Motion data: {len(motndf) if not motndf.empty else 'EMPTY'} rows")
                except Exception as e:
                    print(f"Error downloading motion data: {str(e)}")
                    motndf = pd.DataFrame()
                
                try:
                    sensorsdf = download_blob_to_dataframe(container_client, 'sensorsmetadata.csv')
                    print(f"Sensors metadata: {len(sensorsdf) if not sensorsdf.empty else 'EMPTY'} rows")
                except Exception as e:
                    print(f"Error downloading sensors metadata: {str(e)}")
                    sensorsdf = pd.DataFrame()
                
                try:
                    assetsdf = download_blob_to_dataframe(container_client, 'rdf.csv')
                    print(f"Assets data: {len(assetsdf) if not assetsdf.empty else 'EMPTY'} rows")
                except Exception as e:
                    print(f"Error downloading assets data: {str(e)}")
                    assetsdf = pd.DataFrame()
                
                # Merge assets data with sensors data
                if not sensorsdf.empty and not assetsdf.empty:
                    if "AssetId" in sensorsdf.columns and "AssetId" in assetsdf.columns:
                        try:
                            sensorsdf = pd.merge(sensorsdf, assetsdf[["AssetId", "Description"]], on="AssetId", how="left")
                            print("Successfully merged assets data with sensors data")
                        except Exception as e:
                            print(f"Error merging assets data: {str(e)}")
                
                return tempdf, battdf, vibdf, motndf, sensorsdf
            else:
                logging.error("STORAGE_CONNECTION_STRING environment variable not set")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading data from Azure Blob Storage {str(e)}")
                
            # Fallback to local files if configured for testing
            if os.environ.get('USE_LOCAL_FILES', 'False').lower() == 'true':
                logger.info("Falling back to local files")
                try:
                    tempdf = pd.read_csv('data/temperaturedata.csv')
                    battdf = pd.read_csv('data/batterydata.csv')
                    vibdf = pd.read_csv('data/vibrationdata.csv')
                    motndf = pd.read_csv('data/motiondata.csv')
                    sensorsdf = pd.read_csv('data/sensorsmetadata.csv')
                    assetsdf = pd.read_csv('data/rdf.csv', encoding='latin1')
                        
                    # Merge assets data with sensors data
                    if "AssetId" in sensorsdf.columns and "AssetId" in assetsdf.columns:
                        sensorsdf = pd.merge(sensorsdf, assetsdf[["AssetId", "Description"]], on="AssetId", how="left")
                        
                    return tempdf, battdf, vibdf, motndf, sensorsdf
                except Exception as elocal:
                    logger.error(f"Error loading local files {str(elocal)}")
                
            # Return empty dataframes as a last resort
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        

    def download_blob_to_dataframe(container_client, blob_name):
        """Download a blob and convert to DataFrame"""
        try:
            # Get blob client
            blob_client = container_client.get_blob_client(blob_name)
            
            # Download blob
            download_stream = blob_client.download_blob()
            
            # Convert to DataFrame
            if blob_name.endswith('.csv'):
                if blob_name == 'rdf.csv':
                    df = pd.read_csv(io.BytesIO(download_stream.readall()), encoding='latin1')
                else:
                    df = pd.read_csv(io.BytesIO(download_stream.readall()))
            else:
                # Handle other file types as needed
                raise ValueError(f"Unsupported file type {blob_name}")
            
            return df
        except Exception as e:
            logger.error(f"Error downloading {blob_name} {str(e)}")
            return pd.DataFrame()

    def add_severity_labels(insights, combined_data):
        """Add severity labels to insights based on anomaly scores and multiple detections"""
        if not insights:
            logger.warning("No insights to label")
            return {}
            
        try:
            # Process short-term insights
            for insight in insights.get('short_term', []):
                sensor_id = insight.get('sensor_id')
                sensor_type = insight.get('sensor_type')
                
                if sensor_id and sensor_type and sensor_type in combined_data:
                    # Safely access sensor data
                    sensor_df = combined_data[sensor_type]
                    if 'sensor_id' in sensor_df.columns:
                        sensor_df = sensor_df[sensor_df['sensor_id'] == sensor_id]
                        
                        if not sensor_df.empty:
                            # Count how many detection methods flagged this sensor
                            detection_count = 0
                            if 'statistical_anomaly' in sensor_df.columns and sensor_df['statistical_anomaly'].any():
                                detection_count += 1
                            if 'isolation_forest_anomaly' in sensor_df.columns and sensor_df['isolation_forest_anomaly'].any():
                                detection_count += 1
                            if 'time_series_anomaly' in sensor_df.columns and sensor_df['time_series_anomaly'].any():
                                detection_count += 1
                            
                            # Threshold violations are always critical
                            if 'threshold_violation' in sensor_df.columns and sensor_df['threshold_violation'].any():
                                insight['severity'] = 'critical'
                                insight['multi_detection'] = True
                                
                                # Prefix with CRITICAL if not already
                                if not insight['text'].startswith('CRITICAL'):
                                    insight['text'] = f"CRITICAL: {insight['text']}"
                                continue
                            
                            # Set multi-detection flag
                            insight['multi_detection'] = detection_count >= 2
                            
                            # Default severity based on anomaly score
                            if insight.get('anomaly_score', 0) > 0.8:
                                insight['severity'] = 'critical'
                            elif insight.get('anomaly_score', 0) > 0.6:
                                insight['severity'] = 'concerning'
                            elif insight.get('anomaly_score', 0) > 0.4:
                                insight['severity'] = 'moderate'
                            else:
                                insight['severity'] = 'minor'
                            
                            # Boost severity for multi-detection
                            if detection_count >= 2:
                                # Upgrade severity
                                if insight['severity'] == 'minor':
                                    insight['severity'] = 'moderate'
                                elif insight['severity'] == 'moderate':
                                    insight['severity'] = 'concerning'
                                elif insight['severity'] == 'concerning':
                                    insight['severity'] = 'critical'
            
            # Similar processing for long-term insights
            for insight in insights.get('long_term', []):
                if insight.get('trend_score', 0) > 0.7:
                    insight['severity'] = 'concerning'
                elif insight.get('trend_score', 0) > 0.5:
                    insight['severity'] = 'moderate'
                else:
                    insight['severity'] = 'minor'
            
            # Process relationship insights
            for insight in insights.get('relationships', []):
                # Larger communities are more significant
                community_size = len(insight.get('sensors', []))
                
                if community_size >= 5:
                    insight['severity'] = 'critical'
                elif community_size >= 3:
                    insight['severity'] = 'concerning'
                else:
                    insight['severity'] = 'moderate'
                
                # Check if any sensor in the community has threshold violations
                has_threshold_violation = False
                for sensor_id in insight.get('sensors', []):
                    for sensor_type, df in combined_data.items():
                        if 'sensor_id' in df.columns and 'threshold_violation' in df.columns:
                            sensor_data = df[(df['sensor_id'] == sensor_id) & (df['threshold_violation'] == True)]
                            if not sensor_data.empty:
                                has_threshold_violation = True
                                break
                    
                    if has_threshold_violation:
                        break
                
                # Mark critical if any sensor has threshold violation
                if has_threshold_violation:
                    insight['severity'] = 'critical'
                    # Prefix with CRITICAL if not already
                    if not insight.get('text', '').startswith('CRITICAL'):
                        insight['text'] = f"CRITICAL: {insight.get('text', '')}"
            
            return insights
            
        except Exception as e:
            logger.error(f"Error adding severity labels: {str(e)}")
            # Return original insights without disrupting pipeline
            return insights

    def save_insights(insights):
        """Save insights to Cosmos DB, grouped by account_id"""
        if not insights:
            logger.warning("No insights to save")
            return
            
        try:
            from azure.cosmos import CosmosClient
            
            # Get cosmos connection details
            cosmos_url = os.environ.get('COSMOS_ENDPOINT')
            cosmos_key = os.environ.get('COSMOS_KEY')
            database_name = os.environ.get('COSMOS_DATABASE', 'InsightsDB')
            
            if not cosmos_url or not cosmos_key:
                logger.error("Missing Cosmos DB connection details")
                return
            
            # Initialize Cosmos client
            client = CosmosClient(url=cosmos_url, credential=cosmos_key)
            
            try:
                # Get or create database
                try:
                    database = client.get_database_client(database_name)
                    database.read()  # Check if exists
                except:
                    logger.info(f"Creating database {database_name}")
                    database = client.create_database(database_name)
            except Exception as e:
                logger.error(f"Error accessing/creating Cosmos DB database: {str(e)}")
                return
            
            # Create container for short-term insights
            try:
                try:
                    st_container = database.get_container_client('short-term-insights')
                    st_container.read()  # Check if exists
                except:
                    logger.info("Creating short-term-insights container")
                    st_container = database.create_container(id='short-term-insights', partition_key_path='/account_id')
                
                # Group short-term insights by account_id
                account_st_insights = {}
                for insight in insights.get('short_term', []):
                    account_id = insight.get('account_id', 'unknown')
                    if account_id not in account_st_insights:
                        account_st_insights[account_id] = []
                    account_st_insights[account_id].append(insight)
                
                # Save short-term insights by account
                for account_id, account_insights_list in account_st_insights.items():
                    doc = {
                        'id': f"st_{account_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        'account_id': account_id,
                        'insights': account_insights_list,
                        'timestamp': datetime.now().isoformat()
                    }
                    st_container.upsert_item(doc)
                    logger.info(f"Saved {len(account_insights_list)} short-term insights for account {account_id}")
            except Exception as e:
                logger.error(f"Error saving short-term insights: {str(e)}")
            
            # Create container for long-term insights
            try:
                try:
                    lt_container = database.get_container_client('long-term-insights')
                    lt_container.read()  # Check if exists
                except:
                    logger.info("Creating long-term-insights container")
                    lt_container = database.create_container(id='long-term-insights', partition_key_path='/account_id')
                
                # Group long-term insights by account_id
                account_lt_insights = {}
                for insight in insights.get('long_term', []):
                    account_id = insight.get('account_id', 'unknown')
                    if account_id not in account_lt_insights:
                        account_lt_insights[account_id] = []
                    account_lt_insights[account_id].append(insight)
                
                # Save long-term insights by account
                for account_id, account_insights_list in account_lt_insights.items():
                    doc = {
                        'id': f"lt_{account_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        'account_id': account_id,
                        'insights': account_insights_list,
                        'timestamp': datetime.now().isoformat()
                    }
                    lt_container.upsert_item(doc)
                    logger.info(f"Saved {len(account_insights_list)} long-term insights for account {account_id}")
            except Exception as e:
                logger.error(f"Error saving long-term insights: {str(e)}")
            
            # Create container for relationship insights
            try:
                try:
                    rel_container = database.get_container_client('relationship-insights')
                    rel_container.read()  # Check if exists
                except:
                    logger.info("Creating relationship-insights container")
                    rel_container = database.create_container(id='relationship-insights', partition_key_path='/account_id')
                
                # Group relationship insights by account_id
                account_rel_insights = {}
                
                # Go through each relationship insight
                for insight in insights.get('relationships', []):
                    # Get all sensors in this relationship
                    all_sensors = insight.get('sensors', [])
                    
                    # Find all accounts associated with these sensors
                    account_ids = set()
                    
                    # From short-term insights
                    for st_insight in insights.get('short_term', []):
                        if st_insight.get('sensor_id') in all_sensors:
                            account_id = st_insight.get('account_id', 'unknown')
                            account_ids.add(account_id)
                    
                    # If we didn't find any accounts, try to get it from sensors_df
                    if not account_ids and 'sensors_by_account' in insight:
                        account_ids = set(insight['sensors_by_account'].keys())
                    
                    # If still no accounts, use 'unknown'
                    if not account_ids:
                        account_ids = {'unknown'}
                    
                    # Add this insight to each account's list
                    for account_id in account_ids:
                        if account_id not in account_rel_insights:
                            account_rel_insights[account_id] = []
                        account_rel_insights[account_id].append(insight)
                
                # Save relationship insights by account
                for account_id, account_insights_list in account_rel_insights.items():
                    doc = {
                        'id': f"rel_{account_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        'account_id': account_id,
                        'insights': account_insights_list,
                        'timestamp': datetime.now().isoformat()
                    }
                    rel_container.upsert_item(doc)
                    logger.info(f"Saved {len(account_insights_list)} relationship insights for account {account_id}")
            except Exception as e:
                logger.error(f"Error saving relationship insights: {str(e)}")
                
            logger.info("Successfully saved insights to Cosmos DB")
                
        except Exception as e:
            logger.error(f"Error saving insights to Cosmos DB: {str(e)}")
            traceback.print_exc()

    def save_processed_data(combined_data):
        """Save processed data to blob storage for future reference"""
        if not combined_data:
            logger.warning("No processed data to save")
            return
            
        try:
            from azure.storage.blob import BlobServiceClient
            
            # Get storage connection details
            connection_string = os.environ.get('STORAGE_CONNECTION_STRING')
            container_name = os.environ.get('PROCESSED_CONTAINER', 'processed-data')
            
            if not connection_string:
                logger.error("Missing storage connection string")
                return
            
            # Initialize blob service
            blob_service = BlobServiceClient.from_connection_string(connection_string)
            
            # Get or create container
            try:
                container_client = blob_service.get_container_client(container_name)
                container_client.get_container_properties()  # Check if exists
            except:
                logger.info(f"Creating container {container_name}")
                blob_service.create_container(container_name)
                container_client = blob_service.get_container_client(container_name)
            
            # Save each dataset
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            
            for sensor_type, df in combined_data.items():
                if df is not None and not df.empty:
                    try:
                        # Convert to CSV
                        csv_data = df.to_csv(index=False).encode('utf-8')
                        
                        # Upload to blob storage
                        blob_name = f"{sensor_type}_{timestamp}.csv"
                        blob_client = container_client.get_blob_client(blob_name)
                        blob_client.upload_blob(csv_data, overwrite=True)
                        
                        logger.info(f"Saved processed {sensor_type} data to blob: {blob_name}")
                    except Exception as e:
                        logger.error(f"Error saving {sensor_type} data: {str(e)}")
            
            logger.info("Successfully saved processed data to blob storage")
            
        except Exception as e:
            logger.error(f"Error saving processed data to blob storage: {str(e)}")
            traceback.print_exc()

    def trigger_alerts_for_critical_insights(insights):
        """Trigger alerts for critical insights via Streamlit UI"""
        if not insights:
            logger.warning("No insights to alert")
            return
            
        try:
            # Filter critical insights
            critical_insights = []
            
            for insight in insights.get('short_term', []):
                if insight.get('severity') == 'critical':
                    critical_insights.append(insight)
            
            for insight in insights.get('relationships', []):
                if insight.get('severity') == 'critical':
                    critical_insights.append(insight)
            
            if not critical_insights:
                logger.info("No critical insights to alert")
                return
            
            # Save critical insights to a special container in Cosmos DB for Streamlit to pick up
            from azure.cosmos import CosmosClient
            
            # Get cosmos connection details
            cosmos_url = os.environ.get('COSMOS_ENDPOINT')
            cosmos_key = os.environ.get('COSMOS_KEY')
            database_name = os.environ.get('COSMOS_DATABASE', 'InsightsDB')
            
            if not cosmos_url or not cosmos_key:
                logger.error("Missing Cosmos DB connection details for alerts")
                return
            
            # Initialize Cosmos client
            client = CosmosClient(url=cosmos_url, credential=cosmos_key)
            database = client.get_database_client(database_name)
            
            # Create container for alerts if it doesn't exist
            try:
                try:
                    alerts_container = database.get_container_client('anomaly-alerts')
                    alerts_container.read()  # Check if exists
                except:
                    logger.info("Creating anomaly-alerts container")
                    alerts_container = database.create_container(
                        id='anomaly-alerts',
                        partition_key='/id'
                    )
                    
                # Save alerts
                doc = {
                    'id': f"alert_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    'criticality': 'critical',
                    'insights': critical_insights,
                    'timestamp': datetime.now().isoformat(),
                    'acknowledged': False
                }
                
                alerts_container.upsert_item(doc)
                
                logger.info(f"Triggered alerts for {len(critical_insights)} critical insights")
            except Exception as e:
                logger.error(f"Error creating alert: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error triggering alerts: {str(e)}")
            traceback.print_exc()

    if __name__ == "__main__":
        runpipeline()
    
except Exception as e:
    print(f"CRITICAL ERROR: {str(e)}")
    traceback.print_exc()
    time.sleep(5)  # Give logs time to flush
    sys.exit(1)