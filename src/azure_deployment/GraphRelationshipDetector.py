import pandas as pd
import numpy as np
import logging
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import time
import traceback

class GraphRelationshipDetector:
    """Anomaly relationship detection using graph-based methods.
    
    This detector identifies relationships between anomalies across different sensors
    using graph theory and network analysis. It can identify:
    1. Anomaly propagation patterns
    2. Root cause sensors
    3. Communities of related anomalies
    4. Temporal relationships between anomalies
    """
    
    def __init__(self, config=None, logger=None):
        """Initialize the graph relationship detector"""
        self.default_config = {
            'relationship_discovery': {
                'min_correlation': 0.3,      # Minimum correlation to establish relationship
                'max_time_diff': 60,         # Maximum time difference (minutes) for temporal analysis
                'use_location_data': True,    # Use location data if available
                'use_functional_groups': True, # Use functional groups (e.g., same equipment)
                'min_anomaly_overlap': 0.05   # Minimum anomaly co-occurrence ratio
            },
            'graph_analysis': {
                'gat_layers': 2,             # Number of GAT layers
                'attention_heads': 4,         # Number of attention heads per layer
                'embedding_dim': 64,          # Dimension of sensor embeddings
                'dropout': 0.2,               # Dropout rate
                'batch_size': 1000,           # Batch size for processing
                'use_anomaly_weights': True,  # Use anomaly scores as weights
                'propagation_threshold': 0.3  # Threshold for anomaly propagation
            },
            'memory_optimization': {
                'max_sensors_per_batch': 100,          # Maximum sensors to process in a batch
                'sparse_correlation': True,            # Use sparse correlation matrices
                'use_incremental_updates': True        # Update graph incrementally
            },
            'community_detection': {
                'algorithm': 'louvain', # Community detection algorithm
                'resolution': 1.0, # Resolution parameter for community detection
                'min_community_size': 2, # Minimum size of a community
                'max_communities': 10 # Maximum number of communities to report
            },
            'visualization': {
                'generate_graphs': True, # Generate graph visualizations
                'node_size_factor': 50, # Factor for node sizes
                'edge_width_factor': 2, # Factor for edge widths
                'show_labels': True, # Show sensor labels on graphs
                'color_by_community': True # Color nodes by community
            },
            'account_analysis': {
                'use_account_info': True,     # Use account information if available
                'single_account_boost': 0.2,  # Boost for sensors from same account
                'account_sensor_linking': True # Link sensors based on account data
            }
        }
        
        self.config = config or self.default_config
        self.logger = logger or logging.getLogger('graph_relationship_detector')
        self.graph = None
        self.communities = None
        self.relationship_insights = []
    
    def detect(self, combined_data, sensors_df=None):
        """Detect relationships between anomalies across different sensors"""
        self.logger.info("Starting graph-based relationship detection")
        
        # Validate inputs
        if not combined_data:
            self.logger.warning("No combined data provided for graph relationship detection")
            return []
            
        # Check that each dataframe has required columns
        for sensor_type, df in combined_data.items():
            if df is None or df.empty:
                self.logger.warning(f"Empty dataframe for {sensor_type}")
                continue
                
            if 'sensor_id' not in df.columns:
                self.logger.warning(f"Missing 'sensor_id' column in {sensor_type} data, attempting to rename")
                # Try to rename SensorId to sensor_id if it exists
                if 'SensorId' in df.columns:
                    combined_data[sensor_type] = df.rename(columns={'SensorId': 'sensor_id'})
                else:
                    self.logger.error(f"No suitable 'sensor_id' column found in {sensor_type} data")
        
        try:
            # Create a new graph
            self.graph = nx.Graph()
            
            # Add nodes (sensors) to the graph
            self.add_nodes_from_sensors(sensors_df)
            
            # Add edges (relationships) based on anomalies
            self.add_edges_from_anomalies(combined_data, sensors_df)
            
            # Skip the rest if graph is empty
            if self.graph.number_of_nodes() == 0 or self.graph.number_of_edges() == 0:
                self.logger.warning("Empty graph after adding nodes and edges, returning empty results")
                return []
            
            # Detect communities in the graph
            self.detect_communities()
            
            # Generate insights from the communities
            self.generate_relationship_insights(combined_data, sensors_df)
            
            # Save graph visualization
            try:
                self.save_graph_visualization_to_blob()
            except Exception as e:
                self.logger.warning(f"Error saving graph visualization: {str(e)}")
            
            # Return the relationship insights
            self.logger.info(f"Generated {len(self.relationship_insights)} relationship insights")
            return self.relationship_insights
            
        except Exception as e:
            self.logger.error(f"Error in graph relationship detection: {str(e)}")
            traceback.print_exc()
            # Return empty list if anything fails
            return []
    
    def add_nodes_from_sensors(self, sensors_df):
        """Add nodes to the graph from the sensors dataframe"""
        if sensors_df is None or sensors_df.empty:
            self.logger.warning("No sensor data provided for graph construction")
            return
            
        try:
            # Determine the sensor ID column
            sensor_id_col = None
            if 'sensor_id' in sensors_df.columns:
                sensor_id_col = 'sensor_id'
            elif 'BluetoothAddress' in sensors_df.columns:
                sensor_id_col = 'BluetoothAddress'
            elif 'SensorId' in sensors_df.columns:
                sensor_id_col = 'SensorId'
            
            if not sensor_id_col:
                self.logger.warning("No sensor ID column found in sensors_df")
                return
                
            # Add each sensor as a node
            nodes_added = 0
            for _, sensor in sensors_df.iterrows():
                sensor_id = sensor.get(sensor_id_col)
                if sensor_id and pd.notna(sensor_id):
                    # Add node with sensor attributes
                    node_attrs = {k: v for k, v in sensor.items() if pd.notna(v)}
                    self.graph.add_node(sensor_id, **node_attrs)
                    nodes_added += 1
                    
            self.logger.info(f"Added {nodes_added} nodes to the graph")
            
        except Exception as e:
            self.logger.error(f"Error adding nodes from sensors: {str(e)}")
    
    def add_edges_from_anomalies(self, combined_data, sensors_df):
        """Add edges to the graph based on anomaly relationships"""
        if not combined_data or not self.graph.nodes:
            self.logger.warning("No data or graph nodes available for edge creation")
            return
            
        try:
            # Get configuration
            config = self.config['relationship_discovery']
            min_correlation = config.get('min_correlation', 0.3)
            use_location_data = config.get('use_location_data', True)
            use_functional_groups = config.get('use_functional_groups', True)
            min_anomaly_overlap = config.get('min_anomaly_overlap', 0.05)
            use_account_info = self.config['account_analysis'].get('use_account_info', True)
            
            # Create a dictionary to store anomaly time series for each sensor
            anomaly_series = {}
            
            # Process each sensor type
            for sensor_type, df in combined_data.items():
                if df is None or df.empty:
                    continue
                    
                # Skip if 'sensor_id' column is missing
                if 'sensor_id' not in df.columns:
                    self.logger.warning(f"Missing 'sensor_id' column in {sensor_type} data, skipping")
                    continue
                
                # Get anomaly columns based on sensor type
                anomaly_cols = self.get_anomaly_columns(df)
                
                if not anomaly_cols:
                    self.logger.warning(f"No anomaly columns found in {sensor_type} data")
                    continue
                
                # Process each sensor
                for sensor_id in df['sensor_id'].unique():
                    if sensor_id not in self.graph.nodes:
                        continue
                        
                    # Get sensor data
                    sensor_df = df[df['sensor_id'] == sensor_id]
                    
                    # Create anomaly time series (1 if any anomaly, 0 otherwise)
                    time_col = 'event_time'
                    if time_col not in sensor_df.columns and 'EventTimeUtc' in sensor_df.columns:
                        time_col = 'EventTimeUtc'
                        
                    if time_col in sensor_df.columns:
                        try:
                            # Convert to datetime if not already
                            if not pd.api.types.is_datetime64_any_dtype(sensor_df[time_col]):
                                sensor_df[time_col] = pd.to_datetime(sensor_df[time_col], errors='coerce')
                                
                            # Check if any rows have valid time values
                            if sensor_df[time_col].isna().all():
                                continue
                                
                            # Create mask for any anomaly
                            anomaly_mask = sensor_df[anomaly_cols].any(axis=1)
                            
                            # Store in dictionary
                            anomaly_series[sensor_id] = {
                                'times': sensor_df[time_col].values,
                                'anomalies': anomaly_mask.values,
                                'type': sensor_type
                            }
                        except Exception as e:
                            self.logger.debug(f"Error creating anomaly time series for {sensor_id}: {str(e)}")
            
            # Extract account information from sensors_df if available
            account_sensors = {}
            if use_account_info and sensors_df is not None and not sensors_df.empty:
                # Determine account ID column
                account_col = None
                if 'account_id' in sensors_df.columns:
                    account_col = 'account_id'
                elif 'AccountId' in sensors_df.columns:
                    account_col = 'AccountId'
                
                # Determine sensor ID column
                sensor_id_col = None
                if 'sensor_id' in sensors_df.columns:
                    sensor_id_col = 'sensor_id'
                elif 'BluetoothAddress' in sensors_df.columns:
                    sensor_id_col = 'BluetoothAddress'
                elif 'SensorId' in sensors_df.columns:
                    sensor_id_col = 'SensorId'
                
                # Create account grouping if both columns exist
                if account_col and sensor_id_col:
                    for _, row in sensors_df.iterrows():
                        account_id = row.get(account_col)
                        sensor_id = row.get(sensor_id_col)
                        
                        if pd.notna(account_id) and pd.notna(sensor_id):
                            if account_id not in account_sensors:
                                account_sensors[account_id] = []
                            account_sensors[account_id].append(sensor_id)
                            
                    self.logger.info(f"Found {len(account_sensors)} accounts with sensors")
            
            # Find relationships between sensors based on anomaly patterns
            self.logger.info("Finding relationships between sensors based on anomaly patterns")
            
            # Get all sensor IDs
            sensor_ids = list(anomaly_series.keys())
            
            # Skip if no sensors
            if not sensor_ids:
                self.logger.warning("No sensors with anomaly data found")
                return
            
            # Process sensors in batches to save memory
            max_sensors = self.config['memory_optimization'].get('max_sensors_per_batch', 100)
            
            for i in range(0, len(sensor_ids), max_sensors):
                batch_sensors = sensor_ids[i:i+max_sensors]
                
                for sensor1 in tqdm(batch_sensors, desc=f"Processing relationships batch {i//max_sensors + 1}/{(len(sensor_ids)-1)//max_sensors + 1}"):
                    # Get sensor1 data
                    series1 = anomaly_series.get(sensor1)
                    if not series1:
                        continue
                    
                    # Find sensor1's account
                    sensor1_account = None
                    if account_sensors:
                        for account_id, sensors in account_sensors.items():
                            if sensor1 in sensors:
                                sensor1_account = account_id
                                break
                    
                    # Find relationships with other sensors
                    for sensor2 in sensor_ids:
                        try:
                            # Skip self-relationships
                            if sensor1 == sensor2:
                                continue
                                
                            # Skip if already processed (undirected graph)
                            if self.graph.has_edge(sensor1, sensor2):
                                continue
                                
                            # Get sensor2 data
                            series2 = anomaly_series.get(sensor2)
                            if not series2:
                                continue
                            
                            # Find sensor2's account
                            sensor2_account = None
                            if account_sensors:
                                for account_id, sensors in account_sensors.items():
                                    if sensor2 in sensors:
                                        sensor2_account = account_id
                                        break
                            
                            # Check if same account
                            same_account = (sensor1_account is not None and sensor1_account == sensor2_account)
                            
                            # Calculate relationship
                            relationship = self.calculate_relationship(series1, series2)
                            
                            # Boost for same account
                            if same_account:
                                boost = self.config['account_analysis'].get('single_account_boost', 0.2)
                                
                                if relationship:
                                    relationship['strength'] += boost
                                    relationship['same_account'] = True
                                    relationship['account_id'] = sensor1_account
                                else:
                                    # Minimal relationship for same account
                                    relationship = {
                                        'strength': boost,
                                        'same_account': True,
                                        'account_id': sensor1_account,
                                        'correlation': 0.0,
                                        'co_occurrence': 0,
                                        'co_occurrence_ratio': 0.0,
                                        'temporal_direction': 0,
                                        'sensor1_type': series1['type'],
                                        'sensor2_type': series2['type']
                                    }
                            
                            # Add edge if relationship is strong enough
                            if relationship and relationship.get('strength', 0) >= min_correlation:
                                self.graph.add_edge(sensor1, sensor2, **relationship)
                                
                        except Exception as e:
                            self.logger.debug(f"Error processing relationship between {sensor1} and {sensor2}: {str(e)}")
            
            # Add edges based on location data if available
            if use_location_data and sensors_df is not None and not sensors_df.empty:
                try:
                    # Find location columns
                    location_cols = [col for col in sensors_df.columns if 'location' in col.lower() or 'latitude' in col.lower() or 'longitude' in col.lower()]
                    
                    if location_cols:
                        self.add_location_based_edges(sensors_df, location_cols[0])
                except Exception as e:
                    self.logger.warning(f"Error adding location-based edges: {str(e)}")
                
            # Add edges based on functional groups if available
            if use_functional_groups and sensors_df is not None and not sensors_df.empty:
                try:
                    # Find group or asset columns
                    group_cols = [col for col in sensors_df.columns if 'group' in col.lower() or 'asset' in col.lower()]
                    
                    if group_cols:
                        self.add_functional_group_edges(sensors_df, group_cols[0])
                except Exception as e:
                    self.logger.warning(f"Error adding functional group edges: {str(e)}")
            
            self.logger.info(f"Added {self.graph.number_of_edges()} edges to the graph")
            
        except Exception as e:
            self.logger.error(f"Error adding edges from anomalies: {str(e)}")
            traceback.print_exc()
    
    def calculate_relationship(self, series1, series2):
        """Calculate the relationship strength between two anomaly time series"""
        try:
            # Get times and anomalies
            times1 = series1.get('times')
            anomalies1 = series1.get('anomalies')
            times2 = series2.get('times')
            anomalies2 = series2.get('anomalies')
            
            # Check inputs
            if times1 is None or anomalies1 is None or times2 is None or anomalies2 is None:
                return None
                
            if len(times1) == 0 or len(anomalies1) == 0 or len(times2) == 0 or len(anomalies2) == 0:
                return None
                
            # Check if times are timestamp objects
            if not np.issubdtype(times1.dtype, np.datetime64) or not np.issubdtype(times2.dtype, np.datetime64):
                self.logger.debug("Time values are not datetime64 type, cannot calculate relationship")
                return None
                
            # Find overlapping time range
            try:
                min_time = max(times1.min(), times2.min())
                max_time = min(times1.max(), times2.max())
            except (TypeError, ValueError) as e:
                self.logger.debug(f"Error finding time range: {str(e)}")
                return None
            
            # Skip if no overlap
            if min_time >= max_time:
                return None
                
            # Filter to overlapping time range
            mask1 = (times1 >= min_time) & (times1 <= max_time)
            mask2 = (times2 >= min_time) & (times2 <= max_time)
            
            times1_overlap = times1[mask1]
            anomalies1_overlap = anomalies1[mask1]
            times2_overlap = times2[mask2]
            anomalies2_overlap = anomalies2[mask2]
            
            # Skip if not enough data
            if len(times1_overlap) < 5 or len(times2_overlap) < 5:
                return None
                
            # Calculate co-occurrence ratio
            anomaly_count1 = np.sum(anomalies1_overlap)
            anomaly_count2 = np.sum(anomalies2_overlap)
            
            # Skip if no anomalies
            if anomaly_count1 == 0 or anomaly_count2 == 0:
                return None
                
            # Create a common timeline by merging and resampling
            try:
                all_times = np.concatenate([times1_overlap, times2_overlap])
                all_times = np.sort(np.unique(all_times))
                
                # Resample anomalies to common timeline
                resampled1 = np.zeros(len(all_times), dtype=bool)
                resampled2 = np.zeros(len(all_times), dtype=bool)
                
                for i, t in enumerate(all_times):
                    # Find closest time in each series
                    idx1 = np.argmin(np.abs(times1_overlap - t))
                    idx2 = np.argmin(np.abs(times2_overlap - t))
                    
                    # Get anomaly status
                    if idx1 < len(anomalies1_overlap):
                        resampled1[i] = anomalies1_overlap[idx1]
                    if idx2 < len(anomalies2_overlap):
                        resampled2[i] = anomalies2_overlap[idx2]
                
                # Calculate co-occurrence
                co_occurrence = np.sum(resampled1 & resampled2)
                
                # Calculate correlation
                try:
                    correlation = np.corrcoef(resampled1, resampled2)[0][1]
                    if np.isnan(correlation):
                        correlation = 0
                except Exception as e:
                    self.logger.debug(f"Error calculating correlation: {str(e)}")
                    correlation = 0
                    
                # Calculate temporal relationship (which comes first)
                temporal_direction = 0
                if np.sum(resampled1) > 0 and np.sum(resampled2) > 0:
                    # Find first anomaly in each series
                    try:
                        first_anomaly1 = np.argmax(resampled1)
                        first_anomaly2 = np.argmax(resampled2)
                        
                        if first_anomaly1 < first_anomaly2:
                            temporal_direction = 1  # series1 anomalies tend to precede series2
                        elif first_anomaly2 < first_anomaly1:
                            temporal_direction = -1  # series2 anomalies tend to precede series1
                    except Exception as e:
                        self.logger.debug(f"Error calculating temporal direction: {str(e)}")
                
                # Calculate overall relationship strength
                min_anomalies = min(anomaly_count1, anomaly_count2)
                co_occurrence_ratio = co_occurrence / min_anomalies if min_anomalies > 0 else 0
                
                # Combine correlation and co-occurrence for overall strength
                strength = 0.7 * abs(correlation) + 0.3 * co_occurrence_ratio
                
                return {
                    'strength': float(strength),
                    'correlation': float(correlation),
                    'co_occurrence': int(co_occurrence),
                    'co_occurrence_ratio': float(co_occurrence_ratio),
                    'temporal_direction': int(temporal_direction),
                    'sensor1_type': str(series1['type']),
                    'sensor2_type': str(series2['type'])
                }
            except Exception as e:
                self.logger.debug(f"Error in timeline creation for relationship: {str(e)}")
                return None
        except Exception as e:
            self.logger.debug(f"Error calculating relationship: {str(e)}")
            return None
        
    def add_location_based_edges(self, sensors_df, location_col):
        """Add edges based on physical proximity of sensors"""
        try:
            # Make sure location column exists
            if location_col not in sensors_df.columns:
                self.logger.warning(f"Location column {location_col} not found in sensors_df")
                return
                
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
                return
                
            # Filter sensors with location data
            location_df = sensors_df[[sensor_id_col, location_col]].dropna()
            
            if location_df.empty:
                self.logger.warning("No location data found")
                return
                
            # Group sensors by location
            try:
                location_groups = location_df.groupby(location_col)[sensor_id_col].apply(list).to_dict()
            except Exception as e:
                self.logger.warning(f"Error grouping sensors by location: {str(e)}")
                return
            
            # Add edges between sensors in the same location
            edges_added = 0
            for location, sensors in location_groups.items():
                if len(sensors) < 2:
                    continue
                    
                for i, sensor1 in enumerate(sensors):
                    for sensor2 in sensors[i+1:]:
                        # Skip if sensors not in graph
                        if sensor1 not in self.graph.nodes or sensor2 not in self.graph.nodes:
                            continue
                            
                        # Check if edge already exists
                        if self.graph.has_edge(sensor1, sensor2):
                            # Update existing edge
                            self.graph[sensor1][sensor2]['same_location'] = True
                            self.graph[sensor1][sensor2]['location'] = location
                            
                            # Boost relationship strength
                            self.graph[sensor1][sensor2]['strength'] += 0.2
                        else:
                            # Add new edge with base strength
                            self.graph.add_edge(
                                sensor1, 
                                sensor2,
                                strength=0.2,
                                same_location=True,
                                location=location
                            )
                            edges_added += 1
                            
            self.logger.debug(f"Added {edges_added} location-based edges")
                
        except Exception as e:
            self.logger.warning(f"Error adding location-based edges: {str(e)}")
        
    def add_functional_group_edges(self, sensors_df, group_col):
        """Add edges based on functional groups (e.g., sensors on same equipment)"""
        try:
            # Make sure group column exists
            if group_col not in sensors_df.columns:
                self.logger.warning(f"Group column {group_col} not found in sensors_df")
                return
                
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
                return
                
            # Filter sensors with group data
            group_df = sensors_df[[sensor_id_col, group_col]].dropna()
            
            if group_df.empty:
                self.logger.warning("No group data found")
                return
                
            # Group sensors by functional group
            try:
                group_groups = group_df.groupby(group_col)[sensor_id_col].apply(list).to_dict()
            except Exception as e:
                self.logger.warning(f"Error grouping sensors by functional group: {str(e)}")
                return
            
            # Add edges between sensors in the same group
            edges_added = 0
            for group, sensors in group_groups.items():
                if len(sensors) < 2:
                    continue
                    
                for i, sensor1 in enumerate(sensors):
                    for sensor2 in sensors[i+1:]:
                        # Skip if sensors not in graph
                        if sensor1 not in self.graph.nodes or sensor2 not in self.graph.nodes:
                            continue
                            
                        # Check if edge already exists
                        if self.graph.has_edge(sensor1, sensor2):
                            # Update existing edge
                            self.graph[sensor1][sensor2]['same_group'] = True
                            self.graph[sensor1][sensor2]['group'] = group
                            
                            # Boost relationship strength
                            self.graph[sensor1][sensor2]['strength'] += 0.2
                        else:
                            # Add new edge with base strength
                            self.graph.add_edge(
                                sensor1,
                                sensor2,
                                strength=0.2,
                                same_group=True,
                                group=group
                            )
                            edges_added += 1
                            
            self.logger.debug(f"Added {edges_added} functional group edges")
                            
        except Exception as e:
            self.logger.warning(f"Error adding functional group edges: {str(e)}")
        
    def detect_communities(self):
        """Detect communities of related sensors in the graph"""
        if not self.graph or self.graph.number_of_nodes() == 0 or self.graph.number_of_edges() == 0:
            self.logger.warning("No graph available for community detection")
            self.communities = {}
            return
            
        try:
            # Get configuration
            config = self.config['community_detection']
            algorithm = config.get('algorithm', 'louvain')
            resolution = config.get('resolution', 1.0)
            min_community_size = config.get('min_community_size', 2)
            
            # Try the requested algorithm with fallbacks
            if algorithm == 'louvain':
                try:
                    # Try using community detection from python-louvain package
                    try:
                        import community as community_louvain
                        partition = community_louvain.best_partition(self.graph, resolution=resolution, weight='strength')
                        
                        # Convert partition to communities
                        communities = {}
                        for node, community_id in partition.items():
                            if community_id not in communities:
                                communities[community_id] = []
                            communities[community_id].append(node)
                            
                        # Filter communities by size
                        self.communities = {k: v for k, v in communities.items() if len(v) >= min_community_size}
                        
                    except ImportError:
                        self.logger.warning("community library not installed, falling back to connected components")
                        # Fall back to connected components
                        components = list(nx.connected_components(self.graph))
                        
                        # Filter by size and convert to dict
                        self.communities = {i: list(comp) for i, comp in enumerate(components) if len(comp) >= min_community_size}
                except Exception as e:
                    self.logger.warning(f"Louvain algorithm failed: {str(e)}")
                    # Fall back to connected components
                    components = list(nx.connected_components(self.graph))
                    
                    # Filter by size and convert to dict
                    self.communities = {i: list(comp) for i, comp in enumerate(components) if len(comp) >= min_community_size}
            else:
                # Default to connected components
                components = list(nx.connected_components(self.graph))
                
                # Filter by size and convert to dict
                self.communities = {i: list(comp) for i, comp in enumerate(components) if len(comp) >= min_community_size}
                
            # Limit number of communities if needed
            max_communities = config.get('max_communities', 10)
            if len(self.communities) > max_communities:
                # Sort communities by size, largest first
                community_sizes = [(cid, len(nodes)) for cid, nodes in self.communities.items()]
                community_sizes.sort(key=lambda x: x[1], reverse=True)
                
                # Keep only the largest communities
                self.communities = {cid: self.communities[cid] for cid, _ in community_sizes[:max_communities]}
                
            # Add community information to nodes
            for community_id, nodes in self.communities.items():
                for node in nodes:
                    if node in self.graph.nodes:
                        self.graph.nodes[node]['community_id'] = community_id
                        
            self.logger.info(f"Detected {len(self.communities)} communities of related sensors")
                
        except Exception as e:
            self.logger.error(f"Error in community detection: {str(e)}")
            # Ensure communities exists even in case of error
            self.communities = {}
            traceback.print_exc()
            
    def generate_relationship_insights(self, combined_data, sensors_df):
        """Generate insights from the detected communities"""
        if not self.communities:
            self.logger.warning("No communities detected for insight generation")
            self.relationship_insights = []
            return
        
        try:
            # Empty the insights list first
            self.relationship_insights = []
            
            # Get sensor to account mapping if available
            sensor_accounts = {}
            device_ids = {}
            
            if sensors_df is not None and not sensors_df.empty:
                # Get account column
                account_col = None
                if 'account_id' in sensors_df.columns:
                    account_col = 'account_id'
                elif 'AccountId' in sensors_df.columns:
                    account_col = 'AccountId'
                
                # Get device ID column
                device_col = None
                if 'DeviceId' in sensors_df.columns:
                    device_col = 'DeviceId'
                elif 'device_id' in sensors_df.columns:
                    device_col = 'device_id'
                
                # Get sensor ID column
                sensor_id_col = None
                if 'sensor_id' in sensors_df.columns:
                    sensor_id_col = 'sensor_id'
                elif 'BluetoothAddress' in sensors_df.columns:
                    sensor_id_col = 'BluetoothAddress'
                elif 'SensorId' in sensors_df.columns:
                    sensor_id_col = 'SensorId'
                
                # Build mappings
                if account_col and sensor_id_col:
                    for _, row in sensors_df.iterrows():
                        sensor_id = row.get(sensor_id_col)
                        account_id = row.get(account_col)
                        
                        if pd.notna(sensor_id) and pd.notna(account_id):
                            sensor_accounts[sensor_id] = account_id
                
                if device_col and sensor_id_col:
                    for _, row in sensors_df.iterrows():
                        sensor_id = row.get(sensor_id_col)
                        device_id = row.get(device_col)
                        
                        if pd.notna(sensor_id) and pd.notna(device_id):
                            device_ids[sensor_id] = device_id
            
            # Process each community
            for community_id, sensors in self.communities.items():
                try:
                    # Group sensors by type
                    sensors_by_type = {}
                    
                    for sensor_id in sensors:
                        # Determine sensor type from combined data
                        sensor_type = None
                        for data_type, df in combined_data.items():
                            if df is not None and not df.empty and 'sensor_id' in df.columns and sensor_id in df['sensor_id'].values:
                                sensor_type = data_type
                                break
                                
                        if sensor_type:
                            if sensor_type not in sensors_by_type:
                                sensors_by_type[sensor_type] = []
                            sensors_by_type[sensor_type].append(sensor_id)
                    
                    # Group sensors by account
                    sensors_by_account = {}
                    for sensor_id in sensors:
                        if sensor_id in sensor_accounts:
                            account_id = sensor_accounts[sensor_id]
                            if account_id not in sensors_by_account:
                                sensors_by_account[account_id] = []
                            sensors_by_account[account_id].append(sensor_id)
                    
                    # Calculate community strength
                    community_strength = 0.0
                    edge_count = 0
                    
                    for i, sensor1 in enumerate(sensors):
                        for sensor2 in sensors[i+1:]:
                            if self.graph.has_edge(sensor1, sensor2):
                                edge_data = self.graph[sensor1][sensor2]
                                community_strength += edge_data.get('strength', 0)
                                edge_count += 1
                    
                    avg_strength = community_strength / edge_count if edge_count > 0 else 0
                    
                    # Find potential root cause
                    # Try different centrality measures
                    try:
                        # Start with degree centrality (simplest)
                        subgraph = self.graph.subgraph(sensors)
                        centrality = nx.degree_centrality(subgraph)
                        
                        # Sort sensors by centrality
                        sorted_sensors = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
                        
                        if sorted_sensors:
                            root_sensor = sorted_sensors[0][0]
                        else:
                            # Fallback if no centrality
                            root_sensor = sensors[0] if sensors else None
                        
                    except Exception as e:
                        self.logger.debug(f"Error calculating centrality: {str(e)}")
                        root_sensor = sensors[0] if sensors else None
                    
                    if not root_sensor:
                        continue  # Skip if no root cause can be determined
                    
                    # Get device ID for root sensor
                    root_device = device_ids.get(root_sensor, root_sensor)
                    
                    # Determine root sensor type
                    root_type = None
                    for data_type, df in combined_data.items():
                        if df is not None and not df.empty and 'sensor_id' in df.columns and root_sensor in df['sensor_id'].values:
                            root_type = data_type
                            break
                    
                    # Check if root sensor has threshold violations
                    root_has_threshold = False
                    if root_type and root_sensor:
                        df = combined_data[root_type]
                        if df is not None and not df.empty:
                            if 'sensor_id' in df.columns and 'threshold_violation' in df.columns:
                                sensor_data = df[(df['sensor_id'] == root_sensor) & (df['threshold_violation'] == True)]
                                root_has_threshold = not sensor_data.empty
                    
                    # Generate text for relationship insight
                    insight_text = self._generate_insight_text(
                        community_id=community_id,
                        sensors=sensors,
                        sensors_by_type=sensors_by_type,
                        sensors_by_account=sensors_by_account,
                        root_sensor=root_sensor,
                        root_device=root_device,
                        root_type=root_type,
                        root_has_threshold=root_has_threshold,
                        avg_strength=avg_strength
                    )
                    
                    # Create insight object
                    insight = {
                        'community_id': community_id,
                        'size': len(sensors),
                        'sensors': sensors,
                        'sensors_by_type': sensors_by_type,
                        'sensors_by_account': sensors_by_account,
                        'avg_propagation_score': float(avg_strength),
                        'potential_root_cause': {
                            'sensor_id': root_sensor,
                            'device_id': root_device,
                            'sensor_type': root_type,
                            'has_threshold_violation': root_has_threshold
                        },
                        'text': insight_text
                    }
                    
                    # Add insight
                    self.relationship_insights.append(insight)
                    
                except Exception as e:
                    self.logger.warning(f"Error generating insight for community {community_id}: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error generating relationship insights: {str(e)}")
            traceback.print_exc()
            # Ensure relationship_insights exists even in case of error
            self.relationship_insights = []
    
    def _generate_insight_text(self, community_id, sensors, sensors_by_type, sensors_by_account, 
                            root_sensor, root_device, root_type, root_has_threshold, avg_strength):
        """Generate descriptive text for a relationship insight"""
        try:
            # Start with basic description
            text = f"Detected a related group of {len(sensors)} sensors with anomalies. "
            
            # Add sensor type information
            if len(sensors_by_type) > 1:
                type_counts = []
                for sensor_type, sensors_list in sensors_by_type.items():
                    type_counts.append(f"{len(sensors_list)} {sensor_type}")
                text += f"This group includes {', '.join(type_counts)} sensors. "
                
            # Add account information
            if len(sensors_by_account) == 1:
                account_id = list(sensors_by_account.keys())[0]
                text += f"All sensors belong to account {account_id}. "
            elif len(sensors_by_account) > 1:
                text += f"These sensors span {len(sensors_by_account)} different accounts. "
                
            # Add root cause information
            if root_sensor and root_type:
                text += f"The potential root cause appears to be {root_type} sensor {root_device}. "
                
                # Highlight threshold violations
                if root_has_threshold:
                    text += f"This root sensor has CRITICAL threshold violations that require immediate attention. "
                    
            # Add relationship strength information
            if avg_strength > 0.7:
                text += f"The anomaly patterns show strong correlation (strength: {avg_strength:.2f}). "
            elif avg_strength > 0.5:
                text += f"The anomaly patterns show moderate correlation (strength: {avg_strength:.2f}). "
            else:
                text += f"The anomaly patterns show some correlation (strength: {avg_strength:.2f}). "
                
            # Add recommendation
            text += "Recommend investigating these sensors as a group rather than individually."
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error generating insight text: {str(e)}")
            return f"Group of {len(sensors)} related sensors showing correlated anomalies."
    
    def get_anomaly_columns(self, df):
        """Get anomaly columns in the dataframe"""
        if df is None or df.empty:
            return []
            
        anomaly_cols = []
        
        # Check for standardized column names
        if 'statistical_anomaly' in df.columns:
            anomaly_cols.append('statistical_anomaly')
            
        if 'isolation_forest_anomaly' in df.columns:
            anomaly_cols.append('isolation_forest_anomaly')
            
        if 'time_series_anomaly' in df.columns:
            anomaly_cols.append('time_series_anomaly')
            
        if 'threshold_violation' in df.columns:
            anomaly_cols.append('threshold_violation')
            
        # If no standardized columns, look for any with "anomaly" in the name
        if not anomaly_cols:
            anomaly_cols = [col for col in df.columns if 'anomaly' in col.lower() or 'violation' in col.lower()]
            
        return anomaly_cols
    
    def get_results(self):
        """Get a summary of the graph-based relationship analysis"""
        return {
            'communities': self.communities,
            'relationship_insights': self.relationship_insights,
            'graph_stats': {
                'nodes': self.graph.number_of_nodes() if self.graph else 0,
                'edges': self.graph.number_of_edges() if self.graph else 0,
                'communities': len(self.communities) if self.communities else 0
            }
        }
        
    def visualize_graph(self):
        """Generate a visualization of the sensor relationship graph"""
        if not self.graph or self.graph.number_of_nodes() == 0 or not self.communities:
            self.logger.warning("No graph or communities available for visualization")
            return None
            
        try:
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Try with different layouts
            try:
                pos = nx.spring_layout(self.graph, k=0.3, iterations=50, seed=42)
            except Exception as e:
                self.logger.debug(f"Spring layout failed: {str(e)}")
                try:
                    pos = nx.kamada_kawai_layout(self.graph)
                except:
                    # Fall back to simple layout
                    pos = {}
                    for i, node in enumerate(self.graph.nodes):
                        angle = 2 * np.pi * i / len(self.graph.nodes)
                        pos[node] = np.array([np.cos(angle), np.sin(angle)])
            
            # Setup node colors by community
            node_colors = []
            
            if self.config['visualization'].get('color_by_community', True):
                # Create color map
                cmap = plt.cm.get_cmap('tab10', max(10, len(self.communities)))
                
                # Assign colors to nodes based on community
                for node in self.graph.nodes:
                    community_id = self.graph.nodes[node].get('community_id')
                    if community_id is not None:
                        # Convert community_id to int index for cmap
                        color_idx = int(community_id) % 10  # Use modulo to avoid index errors
                        node_colors.append(cmap(color_idx))
                    else:
                        node_colors.append('lightgray')
            else:
                # Use a single color for all nodes
                node_colors = ['skyblue'] * len(self.graph.nodes)
                
            # Node sizes based on degree
            node_size_factor = self.config['visualization'].get('node_size_factor', 50)
            node_sizes = [node_size_factor * (1 + self.graph.degree(node)) for node in self.graph.nodes]
            
            # Edge widths based on relationship strength
            edge_width_factor = self.config['visualization'].get('edge_width_factor', 2)
            edge_widths = [edge_width_factor * self.graph[u][v].get('strength', 0.1) for u, v in self.graph.edges]
            
            # Draw the network
            nx.draw_networkx_nodes(self.graph, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
            nx.draw_networkx_edges(self.graph, pos, width=edge_widths, alpha=0.5)
            
            # Add labels if configured
            if self.config['visualization'].get('show_labels', True):
                # If many nodes, only label a few
                if len(self.graph.nodes) > 20:
                    # Get nodes with highest degree
                    top_nodes = sorted(self.graph.degree, key=lambda x: x[1], reverse=True)[:10]
                    top_node_ids = [n[0] for n in top_nodes]
                    
                    # Create labels dict with only top nodes
                    labels = {node: str(node) for node in top_node_ids}
                else:
                    labels = {node: str(node) for node in self.graph.nodes}
                    
                nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=8, font_family='sans-serif')
                
            # Remove axes
            plt.axis('off')
            
            # Add title
            plt.title('Sensor Anomaly Relationship Graph', size=15)
            plt.tight_layout()
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            
            # Convert to base64
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            # Close plot to release memory
            plt.close()
            
            return img_str
            
        except Exception as e:
            self.logger.error(f"Error generating graph visualization: {str(e)}")
            if 'plt' in locals():
                plt.close()
            return None
            
    def save_graph_visualization_to_blob(self):
        """Save graph visualization to blob storage"""
        try:
            from azure.storage.blob import BlobServiceClient
            
            # Skip if no graph
            if not self.graph or not self.communities:
                self.logger.warning("No graph visualization to save")
                return
                
            # Get storage connection details
            connection_string = os.environ.get('STORAGE_CONNECTION_STRING')
            container_name = os.environ.get('PROCESSED_CONTAINER', 'processed-data')
            
            if not connection_string:
                self.logger.error("Missing storage connection string")
                return
            
            # Initialize blob service
            blob_service = BlobServiceClient.from_connection_string(connection_string)
            container_client = blob_service.get_container_client(container_name)
            
            # Create visualization
            img_str = self.visualize_graph()
            if not img_str:
                self.logger.warning("Failed to create graph visualization")
                return
                
            # Convert base64 string to bytes
            img_bytes = base64.b64decode(img_str)
            
            # Upload to blob storage
            timestamp = time.strftime('%Y%m%d%H%M%S')
            blob_name = f"visualizations/relationship_graph_{timestamp}.png"
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(img_bytes, overwrite=True)
            
            self.logger.info(f"Saved graph visualization to blob: {blob_name}")
            
        except Exception as e:
            self.logger.error(f"Error saving graph visualization to blob: {str(e)}")