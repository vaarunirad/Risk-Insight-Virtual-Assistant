import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import io
import logging
import json
import os
import warnings
from datetime import datetime, timedelta
from azure.cosmos import CosmosClient
from azure.storage.blob import BlobServiceClient
from openai import OpenAI

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cogorisk_insights")

# Set page config for Streamlit
st.set_page_config(
    page_title="CogoRisk Insights", 
    page_icon="🔍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for color-blind friendly visualization
st.markdown("""
<style>
    .critical { color: #D55E00; font-weight: bold; }
    .concerning { color: #E69F00; font-weight: bold; }
    .moderate { color: #0072B2; font-weight: bold; }
    .minor { color: #56B4E9; font-weight: bold; }
    
    .alert-card {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .critical-alert { background-color: rgba(213, 94, 0, 0.2); border-left: 5px solid #D55E00; }
    .concerning-alert { background-color: rgba(230, 159, 0, 0.2); border-left: 5px solid #E69F00; }
    .moderate-alert { background-color: rgba(0, 114, 178, 0.15); border-left: 5px solid #0072B2; }
    .minor-alert { background-color: rgba(86, 180, 233, 0.15); border-left: 5px solid #56B4E9; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #E6F0FF;
        border-bottom: 2px solid #4B83EC;
    }
</style>
""", unsafe_allow_html=True)

# Initialize connection to Azure services
@st.cache_resource
def init_connections():
    """Initialize and cache connections to Azure Cosmos DB and Blob Storage"""
    try:
        # Specific credentials from the integration document
        cosmos_endpoint = cosmos_endpoint = "YOUR_COSMOS_ENDPOINT"
        # Cosmos DB key for authentication
        cosmos_key = "YOUR_COSMOS_KEY"
        
        # Storage connection string for cogosensordata
        storage_connection_string = "DefaultEndpointsProtocol=https;EndpointSuffix=core.windows.net;AccountName=cogosensordata;AccountKey=YOUR_STORAGE_KEY"
        
        # Connect to Cosmos DB
        cosmos_client = CosmosClient(
            url=cosmos_endpoint,
            credential=cosmos_key
        )
        
        # Get database client
        database = cosmos_client.get_database_client("InsightsDB")
        
        # Get container clients
        short_term_container = database.get_container_client("short-term-insights")
        long_term_container = database.get_container_client("long-term-insights")
        relationship_container = database.get_container_client("relationship-insights")
        
        # Connect to Blob Storage
        blob_service = BlobServiceClient.from_connection_string(storage_connection_string)
        
        # Get blob containers
        raw_container = blob_service.get_container_client("raw-data")
        processed_container = blob_service.get_container_client("processed-data")
        
        return {
            "cosmos_client": cosmos_client,
            "database": database,
            "short_term_container": short_term_container,
            "long_term_container": long_term_container,
            "relationship_container": relationship_container,
            "blob_service": blob_service,
            "raw_container": raw_container,
            "processed_container": processed_container
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize connections: {str(e)}")
        st.error(f"Failed to initialize Azure connections: {str(e)}")
        return None

# Function to load raw data from blob storage
def load_raw_data(container_name, file_name, blob_service):
    """Load data from Azure Blob Storage with direct bytes handling"""
    try:
        logger.info(f"Loading raw data file: {file_name} from {container_name}")
        
        # Get the container client
        raw_container = blob_service.get_container_client(container_name)
        
        # Check if blob exists
        blob_client = raw_container.get_blob_client(file_name)
        
        try:
            # This will raise an error if the blob doesn't exist
            blob_client.get_blob_properties()
        except Exception as e:
            logger.warning(f"File {file_name} not found in {container_name} container: {str(e)}")
            return pd.DataFrame()
        
        # Download the blob without reading it yet
        try:
            # Download blob
            download_stream = blob_client.download_blob()
            content_bytes = download_stream.readall()
            
            if file_name.endswith('.csv'):
                try:
                    # For rdf.csv, we know it needs latin1 encoding
                    encoding = 'latin1' if file_name == 'rdf.csv' else 'utf-8'
                    
                    # Use BytesIO to pass bytes directly to read_csv
                    # This avoids encoding/decoding issues
                    df = pd.read_csv(io.BytesIO(content_bytes), encoding=encoding, on_bad_lines='skip')
                    
                    # Process timestamps if they exist
                    if 'EventTimeUtc' in df.columns:
                        df['EventTimeUtc'] = pd.to_datetime(df['EventTimeUtc'], errors='coerce')
                        df['event_time'] = df['EventTimeUtc']
                    
                    # Rename common columns for consistency
                    if 'SensorId' in df.columns and 'sensor_id' not in df.columns:
                        df['sensor_id'] = df['SensorId']
                    if 'BluetoothAddress' in df.columns and 'sensor_id' not in df.columns:
                        df['sensor_id'] = df['BluetoothAddress']
                    
                    logger.info(f"Successfully loaded {file_name} with {len(df)} rows")
                    return df
                    
                except Exception as specific_e:
                    # If direct approach failed, try the 'python' engine which handles some problematic files better
                    try:
                        logger.warning(f"Standard parser failed, trying python engine: {str(specific_e)}")
                        df = pd.read_csv(io.BytesIO(content_bytes), encoding=encoding, engine='python')
                        
                        # Process the dataframe as before
                        if 'EventTimeUtc' in df.columns:
                            df['EventTimeUtc'] = pd.to_datetime(df['EventTimeUtc'], errors='coerce')
                            df['event_time'] = df['EventTimeUtc']
                        
                        if 'SensorId' in df.columns and 'sensor_id' not in df.columns:
                            df['sensor_id'] = df['SensorId']
                        if 'BluetoothAddress' in df.columns and 'sensor_id' not in df.columns:
                            df['sensor_id'] = df['BluetoothAddress']
                        
                        logger.info(f"Successfully loaded {file_name} with python engine")
                        return df
                    except Exception as python_engine_e:
                        logger.error(f"Both parsers failed for {file_name}: {str(python_engine_e)}")
                        
                        # As a last resort, try to create a simple empty dataframe with minimal columns
                        # This will at least allow the app to continue running
                        logger.info("Creating minimal empty dataframe")
                        empty_df = pd.DataFrame(columns=['sensor_id', 'SensorId', 'DeviceId', 'Description'])
                        return empty_df
            else:
                logger.error(f"Unsupported file type: {file_name}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error downloading blob for {file_name}: {str(e)}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error in load_raw_data for {file_name}: {str(e)}")
        return pd.DataFrame()

# Function to load processed data from blob storage
def load_processed_data(container_name, sensor_type, blob_service):
    """Load most recent processed data from Azure Blob Storage"""
    try:
        logger.info(f"Loading processed data for: {sensor_type}")
        
        # Get container client inside the function
        processed_container = blob_service.get_container_client(container_name)
        
        # List all processed files of this type (they have timestamps in the names)
        blobs = list(processed_container.list_blobs(name_starts_with=f"{sensor_type}_"))
        
        if not blobs:
            logger.warning(f"No processed {sensor_type} data files found")
            return pd.DataFrame()
        
        # Get the most recent one
        latest_blob = sorted(blobs, key=lambda x: x.last_modified, reverse=True)[0]
        
        # Download the data
        blob_client = processed_container.get_blob_client(latest_blob.name)
        downloaded_blob = blob_client.download_blob()
        
        df = pd.read_csv(io.BytesIO(downloaded_blob.readall()))
        
        # Process timestamps if they exist
        if 'event_time' in df.columns:
            df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')
        
        logger.info(f"Successfully loaded processed {sensor_type} data with {len(df)} rows")
        return df
    
    except Exception as e:
        logger.error(f"Error loading processed {sensor_type} data: {str(e)}")
        st.error(f"Error loading processed data: {str(e)}")
        return pd.DataFrame()

# Function to get insights from Cosmos DB
def get_insights(container, account_id=None, start_date=None, end_date=None, critical_only=False):
    """Fetch insights from Cosmos DB with filtering capabilities"""
    try:
        logger.info(f"Getting insights with filters: account_id={account_id}, critical_only={critical_only}")
        
        # Use simpler query approach to avoid parameter type issues
        query_parts = ["SELECT * FROM c WHERE 1=1"]
        
        if account_id and account_id != "All Accounts":
            query_parts.append(f"AND c.account_id = '{account_id}'")
        
        if start_date:
            # Format date properly for SQL query
            formatted_start_date = start_date.isoformat().replace('Z', '')
            query_parts.append(f"AND c.timestamp >= '{formatted_start_date}'")
        
        if end_date:
            formatted_end_date = end_date.isoformat().replace('Z', '')
            query_parts.append(f"AND c.timestamp <= '{formatted_end_date}'")
        
        if critical_only:
            # Simpler query without JSON_QUERY which might not be supported
            query_parts.append("AND CONTAINS(c.insights, 'critical', true)")
        
        # Combine all parts into a single query
        query = " ".join(query_parts)
        logger.info(f"Executing query: {query}")
        
        # Execute query without parameters to avoid type issues
        items = list(container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))

        # Extract insights from items
        all_insights = []
        for item in items:
            for insight in item.get('insights', []):
                # Add timestamp from the document
                insight['timestamp'] = item.get('timestamp')
                # Make sure account_id is included
                if 'account_id' not in insight and 'account_id' in item:
                    insight['account_id'] = item.get('account_id')
                all_insights.append(insight)

        logger.info(f"Retrieved {len(all_insights)} insights")
        return all_insights
    
    except Exception as e:
        logger.error(f"Error fetching insights: {str(e)}")
        st.error(f"Error fetching insights: {str(e)}")
        # Let's print the full traceback for debugging
        import traceback
        logger.error(traceback.format_exc())
        return []

# Function to check for critical alerts to show notifications
def check_critical_alerts(services, account_id=None):
    """Check for any critical alerts across all insight types"""
    try:
        # Get insights from past day only for critical alerts
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        
        # Combine critical insights from all types
        short_term = get_insights(services["short_term_container"], account_id, 
                                yesterday, now, critical_only=True)
        
        long_term = get_insights(services["long_term_container"], account_id, 
                                yesterday, now, critical_only=True)
        
        relationships = get_insights(services["relationship_container"], account_id, 
                                    yesterday, now, critical_only=True)
        
        # Combine all critical insights
        all_critical = short_term + long_term + relationships
        
        return all_critical
    except Exception as e:
        logger.error(f"Error checking for critical alerts: {str(e)}")
        return []

# Dashboard view
def show_dashboard(services, account_id, start_date, end_date, critical_insights):
    """Main dashboard view showing critical alerts and insights summary"""
    st.title("Risk Insights Dashboard")
    st.markdown(f"Showing data from **{start_date.strftime('%b %d, %Y')}** to **{end_date.strftime('%b %d, %Y')}**")

    # Get all insights for the selected time period and account
    short_term_insights = get_insights(services["short_term_container"], account_id, start_date, end_date)
    long_term_insights = get_insights(services["long_term_container"], account_id, start_date, end_date)
    relationship_insights = get_insights(services["relationship_container"], account_id, start_date, end_date)
    
    all_insights = short_term_insights + long_term_insights + relationship_insights
    
    # Load sensors metadata for contextual information
    sensors_df = load_raw_data("raw-data", "sensorsmetadata.csv", services["blob_service"])

    # Display critical alerts section
    st.header("Critical Alerts", divider="red")

    if not critical_insights:
        st.info("No critical alerts found in the selected date range.")
    else:
        # Sort critical insights by timestamp descending (most recent first)
        critical_insights = sorted(critical_insights, key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Show critical alerts
        for idx, insight in enumerate(critical_insights):
            sensor_id = insight.get('sensor_id', 'Unknown')
            device_id = insight.get('device_id', sensor_id)
            sensor_type = insight.get('sensor_type', 'Unknown').capitalize()
            
            # Get description from metadata if available
            description = "Unknown"
            if not sensors_df.empty:
                # Determine the right ID column
                id_col = 'SensorId' if 'SensorId' in sensors_df.columns else 'BluetoothAddress'
                if id_col in sensors_df.columns:
                    sensor_row = sensors_df[sensors_df[id_col] == sensor_id]
                    if not sensor_row.empty and 'Description' in sensor_row.columns:
                        description = sensor_row['Description'].iloc[0]
            
            # Format timestamp for display
            timestamp = insight.get('timestamp', '')
            if timestamp:
                try:
                    timestamp = datetime.fromisoformat(timestamp).strftime('%b %d, %H:%M')
                except:
                    pass
            
            with st.container():
                st.markdown(f"""
                <div class="alert-card critical-alert">
                    <h3>⚠️ {device_id} - {sensor_type} - {description}</h3>
                    <p><strong>{insight.get('text', 'No details available')}</strong></p>
                    <p>Anomaly Score: {insight.get('anomaly_score', 0):.2f} | Detected: {timestamp}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add feedback buttons and action buttons in columns
                col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
                with col1:
                    if st.button("Acknowledge ✓", key=f"ack_{idx}"):
                        st.success("Alert acknowledged!")
                with col2:
                    if st.button("Investigate 🔍", key=f"inv_{idx}"):
                        st.info("Investigation assigned.")
                with col3:
                    if st.button("False Alarm ⚠", key=f"false_{idx}"):
                        st.success("Marked as false alarm.")
                with col4:
                    # Add any additional action buttons or information here
                    pass

    # Display summary statistics
    st.header("Risk Summary", divider="gray")
    
    # Calculate summary statistics
    total_insights = len(all_insights)
    
    # Count by severity
    severities = {
        'critical': sum(1 for i in all_insights if i.get('severity') == 'critical'),
        'concerning': sum(1 for i in all_insights if i.get('severity') == 'concerning'),
        'moderate': sum(1 for i in all_insights if i.get('severity') == 'moderate'),
        'minor': sum(1 for i in all_insights if i.get('severity') == 'minor')
    }
    
    # Count by sensor type
    sensor_types = {
        'temperature': sum(1 for i in all_insights if i.get('sensor_type') == 'temperature'),
        'battery': sum(1 for i in all_insights if i.get('sensor_type') == 'battery'),
        'vibration': sum(1 for i in all_insights if i.get('sensor_type') == 'vibration'),
        'motion': sum(1 for i in all_insights if i.get('sensor_type') == 'motion')
    }
    
    # Count unique sensors and devices
    unique_sensors = set([i.get('sensor_id') for i in all_insights if i.get('sensor_id')])
    unique_devices = set([i.get('device_id') for i in all_insights if i.get('device_id')])
    
    # Count sensors with multiple detections
    multi_detected = sum(1 for i in short_term_insights if i.get('multi_detection', False))
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Anomalies", str(total_insights))
    with col2:
        st.metric("Critical Issues", str(severities['critical']), delta=f"{severities['critical']/max(1, total_insights)*100:.1f}%")
    with col3:
        st.metric("Affected Sensors", str(len(unique_sensors)))
    with col4:
        st.metric("Multi-Detection Anomalies", str(multi_detected), 
                 delta=f"{multi_detected/max(1, len(short_term_insights))*100:.1f}%" if short_term_insights else None)

    # Create visualization for anomalies by severity and type
    col1, col2 = st.columns(2)
    
    with col1:
        # Severity distribution
        st.subheader("Anomaly Severity Distribution")
        
        severity_df = pd.DataFrame({
            'Severity': ['Critical', 'Concerning', 'Moderate', 'Minor'],
            'Count': [severities['critical'], severities['concerning'], severities['moderate'], severities['minor']]
        })
        
        if severity_df['Count'].sum() > 0:
            fig = px.pie(
                severity_df,
                values='Count',
                names='Severity',
                color='Severity',
                color_discrete_map={
                    'Critical': '#D55E00',    # Red-orange (color-blind friendly)
                    'Concerning': '#E69F00',  # Orange
                    'Moderate': '#0072B2',    # Blue
                    'Minor': '#56B4E9'        # Light blue
                },
                hole=0.4
            )
            # Update traces for better formatting
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No anomalies found in the selected time period.")
    
    with col2:
        # Sensor type distribution
        st.subheader("Anomalies by Sensor Type")
        
        type_df = pd.DataFrame({
            'Sensor Type': ['Temperature', 'Battery', 'Vibration', 'Motion'],
            'Count': [sensor_types['temperature'], sensor_types['battery'], 
                     sensor_types['vibration'], sensor_types['motion']]
        })
        
        if type_df['Count'].sum() > 0:
            fig = px.bar(
                type_df,
                x='Sensor Type',
                y='Count',
                color='Sensor Type',
                color_discrete_map={
                    'Temperature': '#0072B2',  # Blue
                    'Battery': '#E69F00',      # Orange
                    'Vibration': '#D55E00',    # Red-orange
                    'Motion': '#56B4E9'        # Light blue
                }
            )
            fig.update_layout(xaxis_title="", yaxis_title="Anomaly Count")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No anomalies found in the selected time period.")

    # Display anomaly trend over time
    st.subheader("Anomaly Trend", divider="gray")
    
    # Prepare data for time series chart
    if all_insights:
        # Extract timestamps and convert to datetime
        timestamps = []
        severities_list = []
        
        for insight in all_insights:
            timestamp = insight.get('timestamp')
            if timestamp:
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                    timestamps.append(timestamp)
                    severities_list.append(insight.get('severity', 'unknown'))
                except:
                    continue
        
        if timestamps:
            # Create dataframe with timestamps and severities
            trend_df = pd.DataFrame({
                'timestamp': timestamps,
                'severity': severities_list
            })
            
            # Group by day and severity
            trend_df['date'] = trend_df['timestamp'].dt.date
            daily_counts = trend_df.groupby(['date', 'severity']).size().reset_index(name='count')
            
            # Create time series chart
            fig = px.line(
                daily_counts,
                x='date',
                y='count',
                color='severity',
                color_discrete_map={
                    'critical': '#D55E00',
                    'concerning': '#E69F00',
                    'moderate': '#0072B2',
                    'minor': '#56B4E9',
                    'unknown': '#999999'
                },
                markers=True
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Number of Anomalies",
                legend_title="Severity"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No time-series data available for the selected period.")
    else:
        st.info("No anomalies found in the selected time period.")

    # Display recent insights section
    st.header("Recent Insights", divider="gray")
    
    # Sort insights by timestamp (most recent first)
    recent_insights = sorted(all_insights, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]

    # Deduplicate insights 
    seen_insights = set()
    unique_insights = []

    for insight in recent_insights:
        insight_key = f"{insight.get('sensor_id', '')}_{insight.get('severity', '')}_{insight.get('text', '')}"

        if insight_key not in seen_insights:
            seen_insights.add(insight_key)
            unique_insights.append(insight)

    unique_insights = unique_insights[:10]
    if not unique_insights:
         st.info("No insights available for the selected time period.")
    else:
        # Create tabs for different severity levels
        tab1, tab2, tab3, tab4 = st.tabs(["All", "Critical", "Concerning", "Moderate/Minor"])

        with tab1:
            display_insights_list(unique_insights, sensors_df)

        with tab2:
            critical_list = [i for i in unique_insights if i.get('severity') == 'critical']
            if critical_list:
                display_insights_list(critical_list, sensors_df)
            else:
                st.info("No critical insights in the selected time period.")

        with tab3:
            concerning_list = [i for i in unique_insights if i.get('severity') == 'concerning']
            if concerning_list:
                display_insights_list(concerning_list, sensors_df)
            else:
                st.info("No concerning insights in the selected time period.")

        with tab4:
            other_list = [i for i in unique_insights if i.get('severity') in ['moderate', 'minor']]
            if other_list:
                display_insights_list(other_list, sensors_df)
            else:
                st.info("No moderate or minor insights in the selected time period.")

def display_insights_list(insights, sensors_df):
    """Helper function to display a list of insights with proper formatting"""
    # Create a set to track seen insights and avoid duplicates
    seen_insights = set()
    
    # Create a cleaned list of insights without duplicates
    unique_insights = []
    
    for insight in insights:
        # Create a unique key for each insight
        insight_text = insight.get('text', '')
        sensor_id = insight.get('sensor_id', '')
        severity = insight.get('severity', '')
        
        # Combined key to identify duplicate insights
        insight_key = f"{sensor_id}_{severity}_{insight_text}"
        
        # Only add if we haven't seen this insight before
        if insight_key not in seen_insights:
            seen_insights.add(insight_key)
            unique_insights.append(insight)
    
    # Now display only the unique insights
    for insight in unique_insights:
        severity = insight.get('severity', 'unknown')
        sensor_id = insight.get('sensor_id', 'Unknown')
        device_id = insight.get('device_id', sensor_id)
        sensor_type = insight.get('sensor_type', 'Unknown').capitalize()

        # Get the insight text and clean it
        
        
        # Create a simple title focusing on the insight type and severity
        expander_title = f"{severity.capitalize()} {sensor_type} Anomaly"
        
        # Create expandable card for each insight
        with st.expander(expander_title, expanded=severity == 'critical'):
            # Use different alert style based on severity
            css_class = f"{severity}-alert"
            
            st.markdown(f"""
            <div class="alert-card {css_class}">
                <p><strong>{insight.get('text', 'No details available')}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show additional details
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Sensor ID:** {sensor_id}")
                st.markdown(f"**Device ID:** {device_id}")
                st.markdown(f"**Type:** {sensor_type}")
                
            with col2:
                if 'anomaly_score' in insight:
                    st.markdown(f"**Anomaly Score:** {insight.get('anomaly_score', 0):.2f}")
                elif 'trend_score' in insight:
                    st.markdown(f"**Trend Score:** {insight.get('trend_score', 0):.2f}")
                
                if 'multi_detection' in insight:
                    st.markdown(f"**Multi-Detection:** {'Yes' if insight.get('multi_detection') else 'No'}")
                
                if 'trend_direction' in insight:
                    direction = insight.get('trend_direction', '')
                    arrow = "↑" if direction == 'increasing' else "↓" if direction == 'decreasing' else ""
                    st.markdown(f"**Trend Direction:** {direction.capitalize()} {arrow}")

# Sensor Explorer view
def show_sensor_explorer(services, account_id, start_date, end_date):
    """Interactive view for exploring sensor data and anomalies"""
    st.title("Sensor Explorer")
    
    # Load sensor metadata
    sensors_df = load_raw_data("raw-data", "sensorsmetadata.csv", services["blob_service"])
    
    # Check if data is available
    if sensors_df.empty:
        st.error("Unable to load sensor metadata")
        return
    
    # Filter by account if needed
    if account_id and account_id != "All Accounts":
        if 'AccountId' in sensors_df.columns:
            sensors_df = sensors_df[sensors_df['AccountId'] == account_id]
        elif 'account_id' in sensors_df.columns:
            sensors_df = sensors_df[sensors_df['account_id'] == account_id]
        else:
            st.warning("Account filtering is not available with the current metadata")
    
    # Get sensor ID column (BluetoothAddress or SensorId)
    sensor_id_col = None
    for col in ['SensorId', 'BluetoothAddress', 'sensor_id']:
        if col in sensors_df.columns:
            sensor_id_col = col
            break
    
    if not sensor_id_col:
        st.error("No sensor ID column found in metadata")
        return
    
    # Create sensor selection interface
    col1, col2 = st.columns(2)
    
    with col1:
        # Determine available sensor types
        if 'SensorType' in sensors_df.columns:
            # Use the SensorType column from metadata
            available_types = sorted(sensors_df['SensorType'].dropna().unique().tolist())
        else:
            # Use default sensor types
            available_types = ["Temperature", "Battery", "Vibration", "Motion"]
        
        selected_type = st.selectbox("Sensor Type", available_types, key="explorer_sensor_type")
    
    with col2:
        # Filter sensors by type if applicable
        if 'SensorType' in sensors_df.columns:
            type_sensors = sensors_df[sensors_df['SensorType'] == selected_type]
        else:
            # If no SensorType column, use all sensors
            type_sensors = sensors_df
        
        # Get sensor options
        sensor_options = sorted(type_sensors[sensor_id_col].dropna().unique().tolist())
        
        if not sensor_options:
            st.warning(f"No {selected_type} sensors found")
            return
        
        selected_sensor = st.selectbox("Select Sensor", sensor_options, key="explorer_sensor_selection")
    
    # Load appropriate sensor data
    # First try processed data
    sensor_data_df = load_processed_data("processed-data", selected_type.lower(), services["blob_service"])
    
    # If processed data is empty or doesn't have this sensor, try raw data
    if sensor_data_df.empty or 'sensor_id' not in sensor_data_df.columns or selected_sensor not in sensor_data_df['sensor_id'].values:
        # Try to load from raw data
        raw_file_name = f"{selected_type.lower()}data.csv"
        sensor_data_df = load_raw_data(services["raw_container"], raw_file_name)
    
    if sensor_data_df.empty:
        st.warning(f"No data available for {selected_sensor}")
        return
    
    # Filter to just this sensor
    if 'sensor_id' in sensor_data_df.columns:
        sensor_data_df = sensor_data_df[sensor_data_df['sensor_id'] == selected_sensor].copy()
    elif 'SensorId' in sensor_data_df.columns:
        sensor_data_df = sensor_data_df[sensor_data_df['SensorId'] == selected_sensor].copy()
    else:
        st.warning(f"Cannot find sensor ID column in data")
        return
    
    # Filter by date range
    time_col = None
    for col in ['EventTimeUtc', 'event_time']:
        if col in sensor_data_df.columns:
            time_col = col
            break
    
    if time_col:
        try:
            # Make sure time column is datetime
            if not pd.api.types.is_datetime64_any_dtype(sensor_data_df[time_col]):
                sensor_data_df[time_col] = pd.to_datetime(sensor_data_df[time_col], errors='coerce')
            
            # Filter by date range
            sensor_data_df = sensor_data_df[(sensor_data_df[time_col] >= start_date) & 
                                           (sensor_data_df[time_col] <= end_date)]
        except Exception as e:
            st.error(f"Error filtering data by date: {str(e)}")
    
    if sensor_data_df.empty:
        st.warning(f"No data found for sensor {selected_sensor} in the selected date range")
        return
    
    # Show sensor information
    display_sensor_info(sensor_data_df, sensors_df, selected_sensor, sensor_id_col, selected_type)
    
    # Show sensor data visualization
    display_sensor_data(sensor_data_df, selected_type)
    
    # Show anomalies for this sensor
    display_sensor_anomalies(services, selected_sensor, selected_type, start_date, end_date, account_id)
    
    # Show raw data table with option to download
    with st.expander("View Raw Data"):
        st.dataframe(sensor_data_df)
        
        # Create a download button
        csv = sensor_data_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{selected_sensor}_{selected_type}_data.csv",
            mime="text/csv"
        )

def display_sensor_info(sensor_data_df, sensors_df, sensor_id, sensor_id_col, sensor_type):
    """Display basic sensor information from metadata"""
    st.subheader("Sensor Information", divider="blue")
    
    # Get the sensor row
    sensor_row = sensors_df[sensors_df[sensor_id_col] == sensor_id]
    
    if sensor_row.empty:
        st.warning("No metadata found for this sensor")
        return
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Get device ID if available
        device_id = None
        for col in ['DeviceId', 'device_id']:
            if col in sensor_row.columns:
                device_id = sensor_row[col].iloc[0]
                break
        
        st.markdown(f"**Sensor ID:** {sensor_id}")
        if device_id:
            st.markdown(f"**Device ID:** {device_id}")
        
        # Get account ID if available
        for col in ['AccountId', 'account_id']:
            if col in sensor_row.columns:
                st.markdown(f"**Account ID:** {sensor_row[col].iloc[0]}")
                break
    
    with col2:
        # Get description if available
        for col in ['Description', 'description']:
            if col in sensor_row.columns and not pd.isna(sensor_row[col].iloc[0]):
                st.markdown(f"**Description:** {sensor_row[col].iloc[0]}")
                break
        
        # Get sensor type if available
        for col in ['SensorType', 'sensor_type']:
            if col in sensor_row.columns and not pd.isna(sensor_row[col].iloc[0]):
                st.markdown(f"**Type:** {sensor_row[col].iloc[0]}")
                break
        
        # Get other useful information
        for col in ['Location', 'location', 'Group', 'group']:
            if col in sensor_row.columns and not pd.isna(sensor_row[col].iloc[0]):
                st.markdown(f"**{col}:** {sensor_row[col].iloc[0]}")
                break

def display_sensor_data(df, sensor_type):
    """Display visualizations of sensor data with anomaly highlighting"""
    st.subheader("Sensor Data Analysis", divider="blue")
    
    # Determine the value column based on sensor type
    value_col = None
    title = None
    y_label = None
    
    if sensor_type.lower() == 'temperature':
        value_cols = ['temperature_celsius', 'TemperatureInCelsius']
        title = "Temperature Readings Over Time"
        y_label = "Temperature (°C)"
    elif sensor_type.lower() == 'battery':
        value_cols = ['battery_voltage', 'BatteryVoltage']
        title = "Battery Voltage Over Time"
        y_label = "Voltage (V)"
    elif sensor_type.lower() == 'vibration':
        value_cols = ['vibration_magnitude', 'MagnitudeRMSInMilliG']
        title = "Vibration Magnitude Over Time"
        y_label = "Magnitude (milliG)"
    elif sensor_type.lower() == 'motion':
        value_cols = ['daily_event_count', 'EventCount', 'max_magnitude']
        title = "Motion Events Over Time"
        y_label = "Count"
    else:
        value_cols = []
    
    # Find the first matching value column that exists
    for col in value_cols:
        if col in df.columns:
            value_col = col
            break
    
    if not value_col:
        st.warning(f"No value column found for {sensor_type} data")
        return
    
    # Determine time column
    time_col = None
    for col in ['EventTimeUtc', 'event_time']:
        if col in df.columns:
            time_col = col
            break
    
    if not time_col:
        st.warning("No time column found in data")
        return
    
    # Check for anomaly columns
    anomaly_cols = []
    for col in ['statistical_anomaly', 'isolation_forest_anomaly', 'time_series_anomaly', 'threshold_violation']:
        if col in df.columns:
            anomaly_cols.append(col)
    
    # Create the line chart
    fig = go.Figure()
    
    # Add the main data line
    fig.add_trace(go.Scatter(
        x=df[time_col],
        y=df[value_col],
        mode='lines+markers',
        name=value_col.replace('_', ' ').title(),
        line=dict(color='#555555', width=2)
    ))
    
    # Add threshold lines if available
    for thresh_col in ['HighAlertThreshold', 'high_threshold', 'LowAlertThreshold', 'low_threshold']:
        if thresh_col in df.columns and not df[thresh_col].isna().all():
            # Get the average threshold value (or max/min for high/low)
            if 'high' in thresh_col.lower() or 'max' in thresh_col.lower():
                thresh_value = df[thresh_col].max()
                line_color = '#D55E00'  # Red-orange
                line_name = "High Threshold"
            else:
                thresh_value = df[thresh_col].min()
                line_color = '#0072B2'  # Blue
                line_name = "Low Threshold"
            
            fig.add_trace(go.Scatter(
                x=[df[time_col].min(), df[time_col].max()],
                y=[thresh_value, thresh_value],
                mode='lines',
                name=line_name,
                line=dict(color=line_color, width=2, dash='dash')
            ))
    
    # Highlight anomalies if available
    for anomaly_col in anomaly_cols:
        # Skip if there are no anomalies
        if anomaly_col not in df.columns or not df[anomaly_col].any():
            continue
        
        # Determine color based on anomaly type
        if 'threshold' in anomaly_col:
            color = '#D55E00'  # Red-orange (critical)
            size = 12
        elif 'forest' in anomaly_col:
            color = '#E69F00'  # Orange (concerning)
            size = 10
        elif 'statistical' in anomaly_col:
            color = '#0072B2'  # Blue (moderate)
            size = 10
        else:
            color = '#56B4E9'  # Light blue (minor)
            size = 10
        
        # Add scatter points for the anomalies
        anomaly_df = df[df[anomaly_col] == True]
        fig.add_trace(go.Scatter(
            x=anomaly_df[time_col],
            y=anomaly_df[value_col],
            mode='markers',
            name=anomaly_col.replace('_', ' ').title(),
            marker=dict(color=color, size=size, symbol='circle')
        ))
    
    # Add score column if available
    for score_col in ['z_score', 'isolation_forest_score', 'time_series_score']:
        if score_col in df.columns and not df[score_col].isna().all():
            # Create secondary y-axis for score
            fig.add_trace(go.Scatter(
                x=df[time_col],
                y=df[score_col],
                mode='lines',
                name=score_col.replace('_', ' ').title(),
                line=dict(color='#999999', width=1, dash='dot'),
                yaxis="y2"
            ))
            
            # Add secondary y-axis configuration
            fig.update_layout(
                yaxis2=dict(
                    title="Anomaly Score",
                    titlefont=dict(color="#999999"),
                    tickfont=dict(color="#999999"),
                    anchor="x",
                    overlaying="y",
                    side="right"
                )
            )
            break
    
    # Configure layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=y_label,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified"
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Show basic statistics
    if value_col in df.columns:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average", f"{df[value_col].mean():.2f}")
        with col2:
            st.metric("Min", f"{df[value_col].min():.2f}")
        with col3:
            st.metric("Max", f"{df[value_col].max():.2f}")
        with col4:
            st.metric("Standard Deviation", f"{df[value_col].std():.2f}")
    
    # Show anomaly statistics if available
    if anomaly_cols:
        st.subheader("Anomaly Statistics")
        
        # Calculate anomaly percentages
        anomaly_stats = []
        for col in anomaly_cols:
            if col in df.columns:
                count = df[col].sum()
                percentage = (count / len(df)) * 100
                anomaly_stats.append({
                    'Type': col.replace('_', ' ').title(),
                    'Count': count,
                    'Percentage': percentage
                })
        
        # Create a dataframe for the stats
        stats_df = pd.DataFrame(anomaly_stats)
        
        # Create bar chart
        if not stats_df.empty and stats_df['Count'].sum() > 0:
            fig = px.bar(
                stats_df,
                x='Type',
                y='Count',
                text='Count',
                color='Type',
                color_discrete_sequence=['#D55E00', '#E69F00', '#0072B2', '#56B4E9'][:len(anomaly_cols)]
            )
            
            fig.update_layout(
                showlegend=False,
                xaxis_title="",
                yaxis_title="Anomaly Count"
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No anomalies detected for this sensor in the selected time range.")

def display_sensor_anomalies(services, sensor_id, sensor_type, start_date, end_date, account_id):
    """Display detected anomalies and insights for the selected sensor"""
    st.subheader("Detected Anomalies", divider="blue")
    
    # Get insights for this specific sensor
    all_insights = get_insights(services["short_term_container"], account_id, start_date, end_date)
    sensor_insights = [i for i in all_insights if i.get('sensor_id') == sensor_id]
    
    # Get long-term insights for this sensor
    long_term_insights = get_insights(services["long_term_container"], account_id, start_date, end_date)
    sensor_long_term = [i for i in long_term_insights if i.get('sensor_id') == sensor_id]
    
    # Get relationship insights involving this sensor
    relationship_insights = get_insights(services["relationship_container"], account_id, start_date, end_date)
    sensor_relationships = [i for i in relationship_insights 
                          if 'sensors' in i and sensor_id in i.get('sensors', [])]
    
    # Create tabs for different insight types
    tabs = []
    
    # Only create tabs for insight types that have data
    if sensor_insights:
        tabs.append("Short-term Anomalies")
    
    if sensor_long_term:
        tabs.append("Long-term Trends")
    
    if sensor_relationships:
        tabs.append("Relationships")
    
    if not tabs:
        st.info("No anomalies or insights detected for this sensor in the selected time range.")
        return
    
    # Create the tabs
    selected_tab = st.radio("Insight Type", tabs, key="sensor_anomaly_tabs")
    
    # Display insights based on selected tab
    if selected_tab == "Short-term Anomalies":
        # Sort by severity and timestamp
        insights = sorted(sensor_insights, 
                        key=lambda x: (0 if x.get('severity') == 'critical' else 
                                      1 if x.get('severity') == 'concerning' else 
                                      2 if x.get('severity') == 'moderate' else 3,
                                     x.get('timestamp', '')), 
                        reverse=True)
        
        if insights:
            for idx, insight in enumerate(insights):
                severity = insight.get('severity', 'unknown')
                
                # Format timestamp for display
                timestamp = insight.get('timestamp', '')
                if timestamp:
                    try:
                        timestamp = datetime.fromisoformat(timestamp).strftime('%b %d, %H:%M')
                    except:
                        pass
                
                # Create card for each insight
                with st.container():
                    # Use different alert style based on severity
                    css_class = f"{severity}-alert"
                    
                    st.markdown(f"""
                    <div class="alert-card {css_class}">
                        <h4>{timestamp} - {severity.capitalize()}</h4>
                        <p><strong>{insight.get('text', 'No details available')}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add technical details
                    with st.expander("Technical Details"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Anomaly Score:** {insight.get('anomaly_score', 0):.2f}")
                            if 'multi_detection' in insight:
                                st.markdown(f"**Multi-Detection:** {'Yes' if insight.get('multi_detection') else 'No'}")
                            
                        with col2:
                            if 'detection_methods' in insight:
                                st.markdown(f"**Detection Methods:** {insight.get('detection_methods')}")
                            if 'anomaly_count' in insight:
                                st.markdown(f"**Anomaly Points:** {insight.get('anomaly_count')} / {insight.get('total_count', 0)}")
                    
                    # Add action buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("Acknowledge", key=f"ack_anomaly_{idx}"):
                            st.success("Anomaly acknowledged!")
                    with col2:
                        if st.button("Investigate", key=f"inv_anomaly_{idx}"):
                            st.info("Investigation assigned.")
                    with col3:
                        if st.button("Mark as False", key=f"false_anomaly_{idx}"):
                            st.success("Marked as false alarm.")
        else:
            st.info("No short-term anomalies detected for this sensor in the selected time range.")
    
    elif selected_tab == "Long-term Trends":
        if sensor_long_term:
            for idx, insight in enumerate(sensor_long_term):
                severity = insight.get('severity', 'unknown')
                
                # Format timestamp for display
                timestamp = insight.get('timestamp', '')
                if timestamp:
                    try:
                        timestamp = datetime.fromisoformat(timestamp).strftime('%b %d, %H:%M')
                    except:
                        pass
                
                # Get trend direction
                direction = insight.get('trend_direction', '')
                arrow = "↑" if direction == 'increasing' else "↓" if direction == 'decreasing' else ""
                
                # Create card for each insight
                with st.container():
                    # Use different alert style based on severity
                    css_class = f"{severity}-alert"
                    
                    st.markdown(f"""
                    <div class="alert-card {css_class}">
                        <h4>{timestamp} - {direction.capitalize()} Trend {arrow}</h4>
                        <p><strong>{insight.get('text', 'No details available')}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add technical details
                    with st.expander("Trend Details"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Trend Score:** {insight.get('trend_score', 0):.2f}")
                            if 'trend_percent' in insight:
                                st.markdown(f"**Change Percentage:** {abs(insight.get('trend_percent', 0)):.2f}%")
                            
                        with col2:
                            if 'days' in insight:
                                st.markdown(f"**Time Period:** {insight.get('days', 0):.1f} days")
                            if 'r_squared' in insight:
                                st.markdown(f"**R-Squared:** {insight.get('r_squared', 0):.3f}")
        else:
            st.info("No long-term trends detected for this sensor in the selected time range.")
    
    elif selected_tab == "Relationships":
        if sensor_relationships:
            for idx, insight in enumerate(sensor_relationships):
                severity = insight.get('severity', 'unknown')
                
                # Format timestamp for display
                timestamp = insight.get('timestamp', '')
                if timestamp:
                    try:
                        timestamp = datetime.fromisoformat(timestamp).strftime('%b %d, %H:%M')
                    except:
                        pass
                
                # Get related sensors
                related_sensors = insight.get('sensors', [])
                sensor_count = len(related_sensors)
                
                # Create card for each insight
                with st.container():
                    # Use different alert style based on severity
                    css_class = f"{severity}-alert"
                    
                    st.markdown(f"""
                    <div class="alert-card {css_class}">
                        <h4>{timestamp} - Relationship with {sensor_count-1} other sensor(s)</h4>
                        <p><strong>{insight.get('text', 'No details available')}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add related sensors
                    with st.expander("Related Sensors"):
                        # Display root cause if available
                        if 'potential_root_cause' in insight:
                            root = insight['potential_root_cause']
                            st.markdown(f"**Root Cause:** {root.get('device_id', root.get('sensor_id', 'Unknown'))} ({root.get('sensor_type', 'Unknown')})")
                        
                        # Group by sensor type
                        if 'sensors_by_type' in insight:
                            st.markdown("**Sensors by Type:**")
                            for sensor_type, sensors in insight['sensors_by_type'].items():
                                if sensors:  # Only show non-empty types
                                    sensor_list = ", ".join(sensors)
                                    st.markdown(f"- **{sensor_type.capitalize()}:** {sensor_list}")
                        
                        # Show correlation strength
                        if 'avg_propagation_score' in insight:
                            st.markdown(f"**Relationship Strength:** {insight.get('avg_propagation_score', 0):.2f}")
        else:
            st.info("No relationships detected for this sensor in the selected time range.")

# Insights view
def show_insights(services, account_id, start_date, end_date):
    """Ultra lightweight insights view that loads instantly"""
    st.title("Risk Insights Analysis")
    
    # Create tabs without loading ANY data initially
    tab1, tab2, tab3 = st.tabs(["Short-term Anomalies", "Long-term Trends", "Relationship Analysis"])
    
    with tab1:
        st.info("Click the button below to load short-term anomaly data")
        load_short_term = st.button("Load Short-term Anomalies", key="load_short_term")
        
        if load_short_term:
            with st.spinner("Loading short-term insights (limited to 10 items)..."):
                try:
                    # Super simple query with strict limit
                    query = f"SELECT TOP 10 * FROM c"
                    if account_id and account_id != "All Accounts":
                        query += f" WHERE c.account_id = '{account_id}'"
                    
                    # Execute query with minimal processing
                    items = list(services["short_term_container"].query_items(
                        query=query,
                        enable_cross_partition_query=True
                    ))
                    
                    # Process with minimal overhead
                    if not items:
                        st.info("No short-term insights found")
                    else:
                        # Just show text versions
                        st.success(f"Found {len(items)} documents with insights")
                        
                        # Extract some insights with strict limits
                        count = 0
                        for item in items:
                            insights = item.get('insights', [])[:2]  # Just get first 2 insights per document
                            for insight in insights:
                                count += 1
                                severity = insight.get('severity', 'unknown')
                                with st.expander(f"{severity.capitalize()} Alert", expanded=severity == 'critical'):
                                    st.write(insight.get('text', 'No details available'))
                        
                        st.info(f"Displaying {count} insights")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with tab2:
        st.info("Click the button below to load trend data")
        load_trends = st.button("Load Long-term Trends", key="load_trends")
        
        if load_trends:
            with st.spinner("Loading trend insights (limited to 5 items)..."):
                try:
                    # Super simple query with strict limit
                    query = f"SELECT TOP 5 * FROM c"
                    if account_id and account_id != "All Accounts":
                        query += f" WHERE c.account_id = '{account_id}'"
                    
                    # Execute query with minimal processing
                    items = list(services["long_term_container"].query_items(
                        query=query,
                        enable_cross_partition_query=True
                    ))
                    
                    # Process with minimal overhead
                    if not items:
                        st.info("No trend insights found")
                    else:
                        # Just show text versions
                        st.success(f"Found {len(items)} documents with trend insights")
                        
                        # Extract some insights with strict limits
                        count = 0
                        for item in items:
                            insights = item.get('insights', [])[:2]  # Just get first 2 insights per document
                            for insight in insights:
                                count += 1
                                sensor_type = insight.get('sensor_type', 'unknown').capitalize()
                                direction = insight.get('trend_direction', 'unknown')
                                arrow = "↑" if direction == 'increasing' else "↓" if direction == 'decreasing' else ""
                                
                                with st.expander(f"{sensor_type} {direction.capitalize()} Trend {arrow}"):
                                    st.write(insight.get('text', 'No details available'))
                        
                        st.info(f"Displaying {count} trend insights")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with tab3:
        st.info("Click the button below to load relationship data")
        load_relationships = st.button("Load Relationship Data", key="load_relationships")
        
        if load_relationships:
            with st.spinner("Loading relationship insights (limited to 3 items)..."):
                try:
                    # Super simple query with strict limit
                    query = f"SELECT TOP 1 * FROM c"
                    if account_id and account_id != "All Accounts":
                        query += f" WHERE c.account_id = '{account_id}'"
                    
                    # Execute query with minimal processing
                    items = list(services["relationship_container"].query_items(
                        query=query,
                        enable_cross_partition_query=True
                    ))
                    
                    # Process with minimal overhead
                    if not items:
                        st.info("No relationship insights found")
                    else:
                        # Just show text versions
                        st.success(f"Found {len(items)} relationship documents")
                        
                        # Extract some insights with strict limits
                        count = 0
                        for item in items:
                            insights = item.get('insights', [])[:1]  # Just get first insight per document
                            for insight in insights:
                                count += 1
                                community_id = insight.get('community_id', 'unknown')
                                sensors = insight.get('sensors', [])
                                
                                with st.expander(f"Community {community_id} - {len(sensors)} sensors"):
                                    st.write(insight.get('text', 'No details available'))
                                    
                                    if 'potential_root_cause' in insight:
                                        root = insight['potential_root_cause']
                                        st.write(f"**Root Cause:** {root.get('device_id', root.get('sensor_id', 'Unknown'))}")
                        
                        st.info(f"Displaying {count} relationship insights")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

def show_short_term_insights(services, sensors_df, account_id, start_date, end_date):
    """Display short-term anomaly insights with filtering and visualization"""
    st.header("Short-term Anomaly Insights")
    
    # Get insights
    insights = get_insights(services["short_term_container"], account_id, start_date, end_date)
    
    if not insights:
        st.info("No short-term insights available for the selected time period.")
        return
    
    # Create filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filter by severity
        severity_options = ["All Severities"] + list(set([i.get('severity', 'unknown').capitalize() 
                                                        for i in insights 
                                                        if 'severity' in i]))
        selected_severity = st.selectbox("Filter by Severity", severity_options, key="st_severity_filter")
    
    with col2:
        # Filter by sensor type
        type_options = ["All Types"] + list(set([i.get('sensor_type', 'unknown').capitalize() 
                                                for i in insights 
                                                if 'sensor_type' in i]))
        selected_type = st.selectbox("Filter by Sensor Type", type_options, key="st_sensor_type_filter")
    
    with col3:
        # Filter by detection method
        detection_options = ["All Methods", "Multi-Detection", "Single Detection"]
        selected_detection = st.selectbox("Filter by Detection Method", detection_options, key="st_detection_filter")
    
    # Apply filters
    filtered_insights = insights.copy()
    
    if selected_severity != "All Severities":
        filtered_insights = [i for i in filtered_insights 
                            if i.get('severity', '').capitalize() == selected_severity]
    
    if selected_type != "All Types":
        filtered_insights = [i for i in filtered_insights 
                            if i.get('sensor_type', '').capitalize() == selected_type]
    
    if selected_detection == "Multi-Detection":
        filtered_insights = [i for i in filtered_insights 
                            if i.get('multi_detection', False)]
    elif selected_detection == "Single Detection":
        filtered_insights = [i for i in filtered_insights 
                            if not i.get('multi_detection', False)]
    
    # Deduplicate insights
    seen_insights = set()
    unique_insights = []
    
    for insight in filtered_insights:
        # Create unique key based on text and sensor
        insight_key = f"{insight.get('sensor_id', '')}_{insight.get('severity', '')}_{insight.get('text', '')}"
        
        if insight_key not in seen_insights:
            seen_insights.add(insight_key)
            unique_insights.append(insight)
    
    # Sort the unique insights by severity and timestamp
    unique_insights = sorted(unique_insights, 
                           key=lambda x: (0 if x.get('severity') == 'critical' else 
                                         1 if x.get('severity') == 'concerning' else 
                                         2 if x.get('severity') == 'moderate' else 3,
                                        x.get('timestamp', '')), 
                           reverse=True)
    
    # Show filter results
    st.markdown(f"**Showing {len(unique_insights)} of {len(insights)} insights**")
    
    if not unique_insights:
        st.info("No insights match the selected filters.")
        return
    
    # Create anomaly distribution visualization
    col1, col2 = st.columns(2)
    
    # Rest of visualization code...
    
    # Display insights
    st.subheader("Anomaly Insights")
    
    # Group insights by device for better overview
    device_insights = {}
    for insight in unique_insights:
        device_id = insight.get('device_id', insight.get('sensor_id', 'Unknown'))
        if device_id not in device_insights:
            device_insights[device_id] = []
        device_insights[device_id].append(insight)
    
    # Sort devices by highest severity
    sorted_devices = sorted(device_insights.items(),
                           key=lambda x: min([0 if i.get('severity') == 'critical' else 
                                             1 if i.get('severity') == 'concerning' else 
                                             2 if i.get('severity') == 'moderate' else 3 
                                             for i in x[1]]))
    
    # Display insights grouped by device
    for device_id, device_insights_list in sorted_devices:
        # Get device type from first insight
        sensor_type = device_insights_list[0].get('sensor_type', 'Unknown').capitalize()
        
        # Get description from metadata if available
        description = "Unknown"
        sensor_id = device_insights_list[0].get('sensor_id', 'Unknown')
        if not sensors_df.empty:
            # Determine the right ID column
            id_col = 'SensorId' if 'SensorId' in sensors_df.columns else 'BluetoothAddress'
            if id_col in sensors_df.columns:
                sensor_row = sensors_df[sensors_df[id_col] == sensor_id]
                if not sensor_row.empty and 'Description' in sensor_row.columns:
                    description = sensor_row['Description'].iloc[0]
        
        # Create expandable section for each device
        with st.expander(f"**{device_id}** - {sensor_type} - {description} ({len(device_insights_list)} anomalies)"):
            # Sort insights by severity and timestamp
            sorted_insights = sorted(device_insights_list, 
                                    key=lambda x: (0 if x.get('severity') == 'critical' else 
                                                  1 if x.get('severity') == 'concerning' else 
                                                  2 if x.get('severity') == 'moderate' else 3,
                                                 x.get('timestamp', '')), 
                                    reverse=True)
            
            for idx, insight in enumerate(sorted_insights):
                severity = insight.get('severity', 'unknown')
                
                # Format timestamp for display
                timestamp = insight.get('timestamp', '')
                if timestamp:
                    try:
                        timestamp = datetime.fromisoformat(timestamp).strftime('%b %d, %H:%M')
                    except:
                        pass
                
                # Create card for each insight
                with st.container():
                    # Use different alert style based on severity
                    css_class = f"{severity}-alert"
                    
                    st.markdown(f"""
                    <div class="alert-card {css_class}">
                        <h4>{timestamp} - {severity.capitalize()}</h4>
                        <p><strong>{insight.get('text', 'No details available')}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add technical details in columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Anomaly Score:** {insight.get('anomaly_score', 0):.2f}")
                        if 'multi_detection' in insight:
                            st.markdown(f"**Multi-Detection:** {'Yes' if insight.get('multi_detection') else 'No'}")
                    
                    with col2:
                        if 'detection_methods' in insight:
                            st.markdown(f"**Detection Methods:** {insight.get('detection_methods')}")
                        if 'anomaly_count' in insight and 'total_count' in insight:
                            percentage = (insight.get('anomaly_count', 0) / insight.get('total_count', 1)) * 100
                            st.markdown(f"**Anomaly Rate:** {percentage:.1f}% ({insight.get('anomaly_count')} / {insight.get('total_count')})")

def show_long_term_insights(services, sensors_df, account_id, start_date, end_date):
    """Display long-term trend insights with filtering and visualization"""
    st.header("Long-term Trend Insights")
    
    # Get insights
    insights = get_insights(services["long_term_container"], account_id, start_date, end_date)
    
    if not insights:
        st.info("No long-term trend insights available for the selected time period.")
        return
    
    # Create filter controls
    col1, col2 = st.columns(2)
    
    with col1:
        # Filter by trend direction
        direction_options = ["All Directions"] + list(set([i.get('trend_direction', '').capitalize() 
                                                          for i in insights 
                                                          if 'trend_direction' in i]))
        selected_direction = st.selectbox("Filter by Trend Direction", direction_options, key="lt_direction_filter")
    
    with col2:
        # Filter by sensor type
        type_options = ["All Types"] + list(set([i.get('sensor_type', '').capitalize() 
                                                for i in insights 
                                                if 'sensor_type' in i]))
        selected_type = st.selectbox("Filter by Sensor Type", type_options, key="lt_sensor_type_filter")
    
    # Apply filters
    filtered_insights = insights.copy()
    
    if selected_direction != "All Directions":
        filtered_insights = [i for i in filtered_insights 
                            if i.get('trend_direction', '').capitalize() == selected_direction]
    
    if selected_type != "All Types":
        filtered_insights = [i for i in filtered_insights 
                            if i.get('sensor_type', '').capitalize() == selected_type]
    
    # Deduplicate insights
    seen_insights = set()
    unique_insights = []
    
    for insight in filtered_insights:
        # Create unique key based on text, sensor, and trend direction
        insight_key = f"{insight.get('sensor_id', '')}_{insight.get('trend_direction', '')}_{insight.get('text', '')}"
        
        if insight_key not in seen_insights:
            seen_insights.add(insight_key)
            unique_insights.append(insight)
    
    # Sort by trend score and timestamp
    unique_insights = sorted(unique_insights, 
                           key=lambda x: (x.get('trend_score', 0), x.get('timestamp', '')), 
                           reverse=True)
    
    # Show filter results
    st.markdown(f"**Showing {len(unique_insights)} of {len(insights)} trend insights**")
    
    if not unique_insights:
        st.info("No insights match the selected filters.")
        return
    
    # Create trend visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Create direction distribution
        direction_counts = {}
        for i in filtered_insights:
            direction = i.get('trend_direction', 'unknown')
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        
        direction_df = pd.DataFrame({
            'Trend Direction': [d.capitalize() for d in direction_counts.keys()],
            'Count': list(direction_counts.values())
        })
        
        if not direction_df.empty:
            fig = px.pie(
                direction_df,
                values='Count',
                names='Trend Direction',
                color='Trend Direction',
                color_discrete_map={
                    'Increasing': '#D55E00',  # Red-orange for increasing (potentially concerning)
                    'Decreasing': '#0072B2',  # Blue for decreasing
                    'Unknown': '#999999'      # Gray for unknown
                },
                title="Trend Direction Distribution"
            )
            # Update traces for better formatting
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Create sensor type distribution
        type_counts = {}
        for i in filtered_insights:
            sensor_type = i.get('sensor_type', 'unknown')
            type_counts[sensor_type] = type_counts.get(sensor_type, 0) + 1
        
        type_df = pd.DataFrame({
            'Sensor Type': [t.capitalize() for t in type_counts.keys()],
            'Count': list(type_counts.values())
        })
        
        if not type_df.empty:
            fig = px.bar(
                type_df,
                x='Sensor Type',
                y='Count',
                color='Sensor Type',
                color_discrete_map={
                    'Temperature': '#0072B2',
                    'Battery': '#E69F00',
                    'Vibration': '#D55E00',
                    'Motion': '#56B4E9',
                    'Unknown': '#999999'
                },
                title="Trends by Sensor Type"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Display trend scatter plot
    st.subheader("Trend Analysis")

    # Extract data for scatter plot
    trend_data = []
    for insight in filtered_insights:
        if 'trend_score' in insight and 'trend_percent' in insight:
            # Add row for scatter plot
            trend_data.append({
                'Device ID': insight.get('device_id', insight.get('sensor_id', 'Unknown')),
                'Sensor Type': insight.get('sensor_type', 'unknown').capitalize(),
                'Trend Score': insight.get('trend_score', 0),
                'Change Percentage': insight.get('trend_percent', 0),
                'Trend Direction': insight.get('trend_direction', 'unknown').capitalize(),
                'Severity': insight.get('severity', 'unknown').capitalize(),
                'Text': insight.get('text', 'No details')
            })

    if trend_data:
        trend_df = pd.DataFrame(trend_data)
        
        # Create scatter plot
        fig = px.scatter(
            trend_df,
            x='Change Percentage',
            y='Trend Score',
            color='Trend Direction',
            symbol='Sensor Type',
            size=abs(trend_df['Change Percentage']) + 1,  # Add 1 to ensure all points visible
            hover_name='Device ID',
            hover_data=['Severity', 'Text'],
            color_discrete_map={
                'Increasing': '#D55E00',  # Red-orange
                'Decreasing': '#0072B2',  # Blue
                'Unknown': '#999999'      # Gray
            },
            title="Trend Score vs Change Percentage"
        )
        
        fig.update_layout(
            xaxis_title="Change Percentage (%)",
            yaxis_title="Trend Score (0-1)",
            xaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='black'),
            hovermode="closest"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.info("""
        **Chart Explanation:**
        - **X-axis:** Percentage change (positive for increasing, negative for decreasing)
        - **Y-axis:** Trend score (higher means more significant trend)
        - **Size:** Magnitude of change (larger circles = bigger change)
        - **Color:** Trend direction (red-orange for increasing, blue for decreasing)
        - **Shape:** Sensor type
        
        Hover over points to see device details and insight text.
        """)
    
    # Display trend insights
    st.subheader("Trend Insights")
    
    # Group insights by device for better overview
    device_insights = {}
    for insight in filtered_insights:
        device_id = insight.get('device_id', insight.get('sensor_id', 'Unknown'))
        if device_id not in device_insights:
            device_insights[device_id] = []
        device_insights[device_id].append(insight)
    
    # Sort devices by trend score
    sorted_devices = sorted(device_insights.items(),
                           key=lambda x: max([i.get('trend_score', 0) for i in x[1]]),
                           reverse=True)
    
    # Display insights grouped by device
    for device_id, device_insights_list in sorted_devices:
        # Get device type from first insight
        sensor_type = device_insights_list[0].get('sensor_type', 'Unknown').capitalize()
        
        # Get description from metadata if available
        description = "Unknown"
        sensor_id = device_insights_list[0].get('sensor_id', 'Unknown')
        if not sensors_df.empty:
            # Determine the right ID column
            id_col = 'SensorId' if 'SensorId' in sensors_df.columns else 'BluetoothAddress'
            if id_col in sensors_df.columns:
                sensor_row = sensors_df[sensors_df[id_col] == sensor_id]
                if not sensor_row.empty and 'Description' in sensor_row.columns:
                    description = sensor_row['Description'].iloc[0]
        
        # Create expandable section for each device
        with st.expander(f"**{device_id}** - {sensor_type} - {description} ({len(device_insights_list)} trends)"):
            # Sort insights by trend score
            sorted_insights = sorted(device_insights_list, 
                                    key=lambda x: x.get('trend_score', 0),
                                    reverse=True)
            
            for idx, insight in enumerate(sorted_insights):
                severity = insight.get('severity', 'unknown')
                
                # Format timestamp for display
                timestamp = insight.get('timestamp', '')
                if timestamp:
                    try:
                        timestamp = datetime.fromisoformat(timestamp).strftime('%b %d, %H:%M')
                    except:
                        pass
                
                # Get trend direction
                direction = insight.get('trend_direction', '')
                arrow = "↑" if direction == 'increasing' else "↓" if direction == 'decreasing' else ""
                
                # Create card for each insight
                with st.container():
                    # Use different alert style based on severity
                    css_class = f"{severity}-alert"
                    
                    st.markdown(f"""
                    <div class="alert-card {css_class}">
                        <h4>{timestamp} - {direction.capitalize()} Trend {arrow}</h4>
                        <p><strong>{insight.get('text', 'No details available')}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add technical details in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"**Trend Score:** {insight.get('trend_score', 0):.2f}")
                    
                    with col2:
                        if 'trend_percent' in insight:
                            st.markdown(f"**Change Percentage:** {abs(insight.get('trend_percent', 0)):.2f}%")
                    
                    with col3:
                        if 'days' in insight:
                            st.markdown(f"**Time Period:** {insight.get('days', 0):.1f} days")

def show_relationship_insights(services, sensors_df, account_id, start_date, end_date):
    """Display relationship insights with simpler graph visualization"""
    st.header("Relationship Analysis")
    
    # Get insights with a limit to avoid processing too much data
    with st.spinner("Loading insights..."):
        try:
            # Use a simplified query with a limit
            insights = get_insights(services["relationship_container"], account_id, start_date, end_date, max_items=20)
            
            if not insights:
                st.info("No relationship insights available for the selected time period.")
                return
        except Exception as e:
            st.error(f"Error loading relationship data: {str(e)}")
            st.info("Try refreshing the page or selecting a different view")
            return
    
    # Display simple text-based list of insights first
    st.subheader("Relationship Insights")
    for insight in insights:
        severity = insight.get('severity', 'unknown')
        sensor_count = len(insight.get('sensors', []))
        
        with st.expander(f"Community of {sensor_count} sensors - {severity.capitalize()}"):
            st.write(f"**{insight.get('text', 'No details available')}**")
            
            # Only show minimal details
            if 'potential_root_cause' in insight:
                root = insight['potential_root_cause']
                root_device = root.get('device_id', root.get('sensor_id', 'Unknown'))
                root_type = root.get('sensor_type', 'unknown').capitalize()
                st.write(f"**Root Cause:** {root_device} ({root_type})")
            
            if 'sensors_by_type' in insight:
                st.write("**Sensors by Type:**")
                for sensor_type, sensors in insight['sensors_by_type'].items():
                    if sensors:  # Only show non-empty types
                        st.write(f"- **{sensor_type.capitalize()}:** {len(sensors)}")
    
    # Offer to show network graph as an option
    if st.button("Generate Network Visualization (may be slow)"):
        with st.spinner("Building network graph..."):
            try:
                # Create a simpler network graph
                G = nx.Graph()
                
                # Add nodes and edges from limited insights
                for insight in insights[:5]:  # Only use first 5 communities max
                    if 'sensors' not in insight:
                        continue
                        
                    sensors = insight.get('sensors', [])[:10]  # Limit to 10 sensors per community
                    
                    # Add nodes
                    for sensor in sensors:
                        G.add_node(sensor)
                    
                    # Add edges (simpler approach)
                    if len(sensors) > 1:
                        # Create a central node for this community
                        community_id = insight.get('community_id', 0)
                        central_node = f"Community-{community_id}"
                        G.add_node(central_node, is_community=True)
                        
                        # Connect all sensors to the central node
                        for sensor in sensors:
                            G.add_edge(central_node, sensor)
                
                # If we have any nodes, create a basic visualization
                if G.number_of_nodes() > 0:
                    # Use a simpler layout algorithm
                    pos = nx.spring_layout(G, seed=42)
                    
                    # Draw network using matplotlib (more stable than plotly for large networks)
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Draw nodes
                    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=50)
                    
                    # Draw edges (with fixed width)
                    nx.draw_networkx_edges(G, pos, ax=ax, width=1)
                    
                    # Draw labels for only a few nodes to avoid clutter
                    # Only label community nodes and first few sensors
                    labels = {n: n for n in G.nodes() if 'Community' in str(n) or n in list(G.nodes())[:5]}
                    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)
                    
                    # Set plot properties
                    plt.axis('off')
                    plt.tight_layout()
                    
                    # Show the plot
                    st.pyplot(fig)
                else:
                    st.info("Not enough relationship data to generate a network graph")
            except Exception as e:
                st.error(f"Error generating relationship graph: {str(e)}")
                st.write("Using a simplified view due to data complexity")

# AI-based Maintenance Advisor
def maintenance_advisor(services):
    """AI-powered maintenance advisor that responds only to what is specifically asked"""
    st.title("Maintenance Advisor")

    # Add a button to clear chat history
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Clear Chat", key="clear_chat_btn"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add a welcome message
        welcome_message = """
        👋 Hello! I'm your Risk Insights Maintenance Advisor. 
        
        You can ask me specific questions about:
        - Troubleshooting sensor issues
        - Maintenance recommendations
        - Interpreting anomalies
        - Best practices for equipment
        
        How can I assist you today?
        """
        st.session_state.messages.append({"role": "assistant", "content": welcome_message})
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Sidebar for sensor context
    st.sidebar.markdown("### Add Context to Your Question")
    
    # Load sensor and asset data for context - keep this part for user selection
    with st.sidebar:
        with st.expander("Sensor Information", expanded=False):
            try:
                # Load sensors metadata
                sensors_df = load_raw_data("raw-data", "sensorsmetadata.csv", services["blob_service"])
                
                # Try to load assets data from rdf.csv
                try:
                    assets_df = load_raw_data("raw-data", "rdf.csv", services["blob_service"])
                    has_assets = not assets_df.empty
                    if has_assets:
                        st.success("Asset data loaded successfully")
                except Exception as e:
                    st.info("Asset data not available. Using sensor information only.")
                    has_assets = False
                    assets_df = pd.DataFrame()
                
                # Get available sensor types
                if 'SensorType' in sensors_df.columns:
                    sensor_types = sorted(sensors_df['SensorType'].dropna().unique().tolist())
                else:
                    sensor_types = ["Temperature", "Battery", "Vibration", "Motion"]
                
                # Let user filter by sensor type
                selected_type = st.selectbox("Filter by Sensor Type", sensor_types, key="advisor_sensor_type")
                
                # Filter sensors by type
                if 'SensorType' in sensors_df.columns:
                    filtered_sensors = sensors_df[sensors_df['SensorType'] == selected_type]
                else:
                    filtered_sensors = sensors_df
                
                # Get sensor ID column
                sensor_id_col = next((col for col in ['SensorId', 'BluetoothAddress', 'sensor_id'] 
                                     if col in filtered_sensors.columns), None)
                
                if sensor_id_col:
                    # Get sensors for the selected type
                    sensor_options = filtered_sensors[sensor_id_col].dropna().unique().tolist()[:10]
                    
                    # Multi-select for sensors
                    selected_sensors = st.multiselect(
                        "Select Sensors for Context:",
                        options=sensor_options,
                        key="advisor_selected_sensors"
                    )
                    
                    # Show selected sensor details
                    if selected_sensors:
                        st.markdown("#### Selected Sensor Details:")
                        for sensor in selected_sensors:
                            sensor_row = filtered_sensors[filtered_sensors[sensor_id_col] == sensor]
                            
                            if not sensor_row.empty:
                                # Get device ID
                                device_id = None
                                for col in ['DeviceId', 'device_id']:
                                    if col in sensor_row.columns:
                                        device_id = sensor_row[col].iloc[0]
                                        break
                                
                                # Get description
                                desc = None
                                for col in ['Description', 'description']:
                                    if col in sensor_row.columns:
                                        desc = sensor_row[col].iloc[0]
                                        break
                                
                                # Display basic info
                                st.markdown(f"**Sensor:** {sensor}")
                                st.markdown(f"**Device:** {device_id or 'Unknown'}")
                                st.markdown(f"**Description:** {desc if pd.notna(desc) else 'nan'}")
                                
                                # Get recent insights for this sensor
                                st.markdown("**Recent Insights:**")
                                
                                insights_found = False
                                try:
                                    insights = []
                                    query = f"SELECT TOP 2 * FROM c WHERE ARRAY_CONTAINS(c.insights, {{sensor_id: '{sensor}'}}, true)"
                                    
                                    for container_name in ["short_term_container", "relationship_container"]:
                                        container = services[container_name]
                                        try:
                                            items = list(container.query_items(
                                                query=query,
                                                enable_cross_partition_query=True
                                            ))
                                            
                                            for item in items:
                                                for insight in item.get('insights', []):
                                                    if insight.get('sensor_id') == sensor:
                                                        insights.append(insight)
                                        except:
                                            pass
                                    
                                    # Display insights as bullet points
                                    if insights:
                                        insights_found = True
                                        # Deduplicate insights
                                        seen_texts = set()
                                        unique_insights = []
                                        
                                        for insight in insights:
                                            text = insight.get('text', '')
                                            if text not in seen_texts:
                                                seen_texts.add(text)
                                                unique_insights.append(insight)
                                        
                                        for insight in unique_insights[:2]:  # Limit to 2 insights
                                            severity = insight.get('severity', 'unknown')
                                            text = insight.get('text', 'No details available')
                                            st.markdown(f"* **{severity.capitalize()}:** Device {device_id or 'Unknown'} ({sensor}) {text}")
                                except:
                                    pass
                                
                                if not insights_found:
                                    st.markdown("* No recent insights available")
                                
                                # Add separator between sensors
                                st.markdown("---")
                                
                        # Button to include context in next question
                        if st.button("Include Context in Question", key="include_context_btn"):
                            # Store selected sensors in session state
                            st.session_state.use_sensor_context = True
                            st.success("Sensor context will be included in your next question")
            except Exception as e:
                st.warning(f"Could not load sensor context: {str(e)}")

    # User input
    if prompt := st.chat_input("Ask a maintenance question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display the new user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Check if we should include sensor context
        use_context = st.session_state.get('use_sensor_context', False)
        enhanced_prompt = prompt
        
        if use_context:
            try:
                # Get context for selected sensors
                selected_sensors = st.session_state.advisor_selected_sensors
                if selected_sensors:
                    sensor_context = []
                    
                    # Load sensor data
                    sensors_df = load_raw_data("raw-data", "sensorsmetadata.csv", services["blob_service"])
                    
                    # Get sensor ID column
                    sensor_id_col = next((col for col in ['SensorId', 'BluetoothAddress', 'sensor_id'] 
                                        if col in sensors_df.columns), None)
                    
                    # Build context for each selected sensor
                    for sensor_id in selected_sensors:
                        sensor_info = f"Sensor ID: {sensor_id}\n"
                        
                        if sensor_id_col:
                            sensor_row = sensors_df[sensors_df[sensor_id_col] == sensor_id]
                            if not sensor_row.empty:
                                # Get device ID
                                for col in ['DeviceId', 'device_id']:
                                    if col in sensor_row.columns and not pd.isna(sensor_row[col].iloc[0]):
                                        sensor_info += f"Device ID: {sensor_row[col].iloc[0]}\n"
                                        break
                                
                                # Get description
                                for col in ['Description', 'description']:
                                    if col in sensor_row.columns and not pd.isna(sensor_row[col].iloc[0]):
                                        sensor_info += f"Description: {sensor_row[col].iloc[0]}\n"
                                        break
                        
                        # Get recent insights
                        try:
                            insights = []
                            query = f"SELECT TOP 1 * FROM c WHERE ARRAY_CONTAINS(c.insights, {{sensor_id: '{sensor_id}'}}, true)"
                            
                            for container_name in ["short_term_container", "relationship_container"]:
                                container = services[container_name]
                                try:
                                    items = list(container.query_items(
                                        query=query,
                                        enable_cross_partition_query=True
                                    ))
                                    
                                    for item in items:
                                        for insight in item.get('insights', []):
                                            if insight.get('sensor_id') == sensor_id:
                                                insights.append(insight)
                                except:
                                    pass
                            
                            if insights:
                                # Add the most recent insight
                                sensor_info += "Recent Insight:\n"
                                insight = insights[0]
                                sensor_info += f"- Severity: {insight.get('severity', 'unknown')}\n"
                                sensor_info += f"- Issue: {insight.get('text', 'No details')}\n"
                        except:
                            pass
                        
                        # Add this sensor context
                        if sensor_info:
                            sensor_context.append(sensor_info)
                    
                    # Add context to prompt if available
                    if sensor_context:
                        enhanced_prompt += "\n\nContext Information:\n"
                        for i, context in enumerate(sensor_context):
                            enhanced_prompt += f"\n--- SENSOR {i+1} ---\n{context}"
                
                # Clear the flag after using it
                st.session_state.use_sensor_context = False
            except Exception as e:
                # Silent fail, just use the original prompt
                pass
        
        # Generate response
        with st.spinner("Generating response..."):
            try:
                import urllib.request
                import json
                
                # Use a system message that emphasizes responding only to what was asked
                system_message = """You are a focused industrial maintenance advisor. Important instructions:

1. ONLY respond to what the user specifically asks about. Do not volunteer information beyond the scope of their question.
2. For simple greetings like "hello", "hi", etc., respond with a brief greeting only - do not provide technical information.
3. Only use sensor context information if:
   - The user specifically asks about the sensor
   - The question relates directly to the issue described in the sensor data
4. Provide practical, focused, and concise responses using technical language appropriate for engineers.
5. Be conversational but avoid unnecessary elaboration beyond what was specifically asked for.

This focused approach is essential to provide the most relevant assistance."""
                
                # Format conversation history for API
                messages = [{"role": "system", "content": system_message}]
                
                # Add previous messages for conversation context
                for msg in st.session_state.messages[-9:]:  # Last 9 messages
                    # Skip the current user message as we'll add the enhanced version
                    if msg["role"] != "user" or msg["content"] != prompt:
                        messages.append({"role": msg["role"], "content": msg["content"]})
                
                # Add the current user message (possibly enhanced with context)
                messages.append({"role": "user", "content": enhanced_prompt})
                
                # Prepare the API request
                api_key = "YOUR_OPENAI_API_KEY"  # Replace with your OpenAI API key
                
                payload = {
                    "model": "gpt-3.5-turbo",
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 800
                }
                
                data = json.dumps(payload).encode('utf-8')
                
                # Create and send request using urllib
                req = urllib.request.Request(
                    "https://api.openai.com/v1/chat/completions",
                    data=data,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}"
                    }
                )
                
                # Make the request and handle response
                with urllib.request.urlopen(req) as response:
                    response_data = json.loads(response.read().decode('utf-8'))
                    response_text = response_data["choices"][0]["message"]["content"]
                    
                    # Add response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.markdown(response_text)
            
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                
                # Add error message to chat history
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Add suggested questions if the conversation is just starting
    if len(st.session_state.messages) <= 2:
        st.markdown("### Suggested Questions:")
        suggested_questions = [
            "What are common causes of high temperature readings in motors?",
            "How do I troubleshoot a battery that's draining too quickly?", 
            "What maintenance is needed for a vibration sensor showing irregular patterns?",
            "What do abnormal temperature fluctuations indicate?",
            "How often should I calibrate sensors?"
        ]
        
        # Display suggestions in columns
        cols = st.columns(2)
        for i, question in enumerate(suggested_questions):
            col = cols[i % 2]
            with col:
                if st.button(question, key=f"suggest_{i}"):
                    # Add to messages and rerun
                    st.session_state.messages.append({"role": "user", "content": question})
                    st.rerun()

# Utility function to get available accounts
def get_available_accounts(services):
    """Get list of available accounts from sensor metadata or insights"""
    accounts = ["All Accounts"]
    
    try:
        # Try to get accounts from Cosmos DB (faster and more reliable)
        query = "SELECT DISTINCT VALUE c.account_id FROM c WHERE c.account_id != null"
        
        # Query from containers
        account_ids = set()
        
        # Short-term container
        try:
            results = list(services["short_term_container"].query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            account_ids.update(results)
        except Exception as e:
            logger.warning(f"Error querying short-term container for accounts: {str(e)}")
        
        # Long-term container
        try:
            results = list(services["long_term_container"].query_items(
                query=query,
                enable_cross_partition_query=True
            ))
            account_ids.update(results)
        except Exception as e:
            logger.warning(f"Error querying long-term container for accounts: {str(e)}")
            
        # Convert to list, sort, and add to accounts
        account_list = sorted([str(a) for a in account_ids if a])
        accounts.extend(account_list)
        
        # If no accounts found, try sensors metadata
        if len(accounts) <= 1:
            # Load sensor metadata
            sensors_df = load_raw_data(services["raw_container"], "sensorsmetadata.csv")
            
            if not sensors_df.empty:
                # Check for account ID column
                account_col = None
                for col in ['AccountId', 'account_id']:
                    if col in sensors_df.columns:
                        account_col = col
                        break
                
                if account_col:
                    # Get unique account IDs
                    unique_accounts = sensors_df[account_col].dropna().unique().tolist()
                    unique_accounts = [str(a) for a in unique_accounts]  # Convert to strings
                    
                    # Sort accounts and add to list if not already there
                    unique_accounts.sort()
                    for acc in unique_accounts:
                        if acc not in accounts:
                            accounts.append(acc)
        
    except Exception as e:
        logger.error(f"Error getting available accounts: {str(e)}")
    
    return accounts

def show_critical_notification(critical_insights):
    """Display a notification for critical alerts"""
    # Count critical alerts by sensor type
    count_by_type = {}
    for insight in critical_insights:
        sensor_type = insight.get('sensor_type', 'unknown')
        count_by_type[sensor_type] = count_by_type.get(sensor_type, 0) + 1
    
    # Create notification message
    critical_count = len(critical_insights)
    
    # Format type counts
    type_text = ", ".join([f"{count} {t.capitalize()}" for t, count in count_by_type.items()])
    
    # Create alert box
    st.warning(
        f"⚠️ **ATTENTION:** {critical_count} critical {'alert' if critical_count == 1 else 'alerts'} detected ({type_text}). Please review the Dashboard for details.",
        icon="⚠️"
    )

def add_sidebar_header():
    """Add logos and title to the sidebar"""
    # Container for logos
    with st.sidebar:
        # Create two columns for the logos
        col1, col2 = st.columns(2)
        
        with col1:
            # Try to display CG logo
            try:
                cogo_logo_path = "./img/logo.png"
                if os.path.exists(cogo_logo_path):
                    st.image(cogo_logo_path, width=100)
                else:
                    st.write("CG")
            except Exception as e:
                st.write("CG")
        
        with col2:
            # Try to display Northeastern logo
            try:
                neu_logo_path = "./img/nucoe-logo-social.png"  # Adjust filename as needed
                if os.path.exists(neu_logo_path):
                    st.image(neu_logo_path, width=100)
                else:
                    st.write("Northeastern University")
            except Exception as e:
                st.write("Northeastern University")
        
        # Add separator
        st.markdown("---")
        
        # Add title
        st.markdown("### Risk Insights Agent")
        st.markdown("#### AI-Powered Maintenance Advisor")
        
        # Add another separator
        st.divider()

def add_footer():
    """Add footer with version information"""
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888888; font-size: 11px;">
        CogoRisk Insights | Version 1.0.0 | &copy; 2025 All Rights Reserved
        </div>
        """, 
        unsafe_allow_html=True
    )

# Main function to run the Streamlit app
def main():
    """Main function to run the CogoRisk Insights application"""
    # Initialize Azure connections
    services = init_connections()
    
    if not services:
        st.error("Failed to initialize Azure connections. Please check your credentials.")
        return
    
    # Add logo and title to the sidebar
    add_sidebar_header()
    
    # Get available accounts from insights data
    accounts = get_available_accounts(services)
    
    # Account selection in sidebar
    selected_account = st.sidebar.selectbox("Select Account", accounts, key="main_account_selector")
    account_id = None if selected_account == "All Accounts" else selected_account
    
    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7), key="main_start_date")
    with col2:
         end_date = st.date_input("End Date", datetime.now(), key="main_end_date")
    
    # Convert to datetime
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    
    # Check for critical alerts
    critical_insights = check_critical_alerts(services, account_id)
    
    # Show critical alert notification if there are any
    if critical_insights:
        show_critical_notification(critical_insights)
    
    # Navigation menu
    page = st.sidebar.radio("Navigation", ["Dashboard", "Sensor Explorer", "Insights", "Maintenance Advisor"], key="main_navigation")
    
    # Display the selected page
    if page == "Dashboard":
        show_dashboard(services, account_id, start_datetime, end_datetime, critical_insights)
    elif page == "Sensor Explorer":
        show_sensor_explorer(services, account_id, start_datetime, end_datetime)
    elif page == "Insights":
        show_insights(services, account_id, start_datetime, end_datetime)
    else:  # Maintenance Advisor
        maintenance_advisor(services)
    
    # Add footer
    add_footer()

if __name__ == "__main__":
    main()