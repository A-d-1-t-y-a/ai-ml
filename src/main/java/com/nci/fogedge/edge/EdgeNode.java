package com.nci.fogedge.edge;

import com.nci.fogedge.network.NetworkManager;
import com.nci.fogedge.utils.MetricsCollector;
import com.nci.fogedge.utils.PerformanceMetrics;

/**
 * Edge Node Interface for the Fog and Edge Computing System
 * 
 * This interface defines the contract for all edge nodes in the system.
 * It provides methods for node lifecycle management, data processing, and communication.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public interface EdgeNode {
    
    /**
     * Get the unique node identifier
     * 
     * @return Node ID
     */
    String getNodeId();
    
    /**
     * Get the node type
     * 
     * @return Node type
     */
    String getNodeType();
    
    /**
     * Get the node location
     * 
     * @return Node location
     */
    String getLocation();
    
    /**
     * Start the edge node
     */
    void start();
    
    /**
     * Stop the edge node
     */
    void stop();
    
    /**
     * Check if the node is running
     * 
     * @return True if node is running
     */
    boolean isRunning();
    
    /**
     * Check if the node is healthy
     * 
     * @return True if node is healthy
     */
    boolean isHealthy();
    
    /**
     * Process data received from IoT devices
     * 
     * @param data Data to process
     * @return Processed data
     */
    String processData(String data);
    
    /**
     * Offload task to cloud services
     * 
     * @param task Task to offload
     * @return True if offloading successful
     */
    boolean offloadTaskToCloud(String task);
    
    /**
     * Get node status
     * 
     * @return Node status information
     */
    String getStatus();
    
    /**
     * Get node metrics
     * 
     * @return Node performance metrics
     */
    PerformanceMetrics getMetrics();
    
    /**
     * Update node configuration
     * 
     * @param config Configuration parameters
     */
    void updateConfiguration(java.util.Map<String, Object> config);
    
    /**
     * Get processing capacity
     * 
     * @return Processing capacity in MB/s
     */
    double getProcessingCapacity();
    
    /**
     * Get storage capacity
     * 
     * @return Storage capacity in MB
     */
    double getStorageCapacity();
    
    /**
     * Get current CPU usage
     * 
     * @return CPU usage percentage (0-100)
     */
    double getCpuUsage();
    
    /**
     * Get current memory usage
     * 
     * @return Memory usage percentage (0-100)
     */
    double getMemoryUsage();
    
    /**
     * Get current storage usage
     * 
     * @return Storage usage percentage (0-100)
     */
    double getStorageUsage();
    
    /**
     * Get network bandwidth usage
     * 
     * @return Bandwidth usage in Mbps
     */
    double getBandwidthUsage();
    
    /**
     * Get energy consumption
     * 
     * @return Energy consumption in watts
     */
    double getEnergyConsumption();
    
    /**
     * Get task processing rate
     * 
     * @return Tasks processed per second
     */
    double getTaskProcessingRate();
    
    /**
     * Get task offloading rate
     * 
     * @return Tasks offloaded per second
     */
    double getTaskOffloadingRate();
    
    /**
     * Get error count
     * 
     * @return Number of errors encountered
     */
    int getErrorCount();
    
    /**
     * Reset node error count
     */
    void resetErrorCount();
    
    /**
     * Get last data processing timestamp
     * 
     * @return Timestamp of last data processing
     */
    long getLastDataProcessingTime();
    
    /**
     * Get last task offloading timestamp
     * 
     * @return Timestamp of last task offloading
     */
    long getLastTaskOffloadingTime();
} 