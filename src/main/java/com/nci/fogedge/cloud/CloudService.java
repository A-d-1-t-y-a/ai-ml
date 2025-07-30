package com.nci.fogedge.cloud;

import com.nci.fogedge.network.NetworkManager;
import com.nci.fogedge.utils.MetricsCollector;
import com.nci.fogedge.utils.PerformanceMetrics;

/**
 * Cloud Service Interface for the Fog and Edge Computing System
 * 
 * This interface defines the contract for all cloud services in the system.
 * It provides methods for service lifecycle management, task processing, and communication.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public interface CloudService {
    
    /**
     * Get the unique service identifier
     * 
     * @return Service ID
     */
    String getServiceId();
    
    /**
     * Get the service type
     * 
     * @return Service type
     */
    String getServiceType();
    
    /**
     * Get the service location
     * 
     * @return Service location
     */
    String getLocation();
    
    /**
     * Start the cloud service
     */
    void start();
    
    /**
     * Stop the cloud service
     */
    void stop();
    
    /**
     * Check if the service is running
     * 
     * @return True if service is running
     */
    boolean isRunning();
    
    /**
     * Check if the service is healthy
     * 
     * @return True if service is healthy
     */
    boolean isHealthy();
    
    /**
     * Process task received from edge nodes
     * 
     * @param task Task to process
     * @return Processing result
     */
    String processTask(String task);
    
    /**
     * Store data in cloud storage
     * 
     * @param data Data to store
     * @return True if storage successful
     */
    boolean storeData(String data);
    
    /**
     * Retrieve data from cloud storage
     * 
     * @param dataId Data identifier
     * @return Retrieved data
     */
    String retrieveData(String dataId);
    
    /**
     * Get service status
     * 
     * @return Service status information
     */
    String getStatus();
    
    /**
     * Get service metrics
     * 
     * @return Service performance metrics
     */
    PerformanceMetrics getMetrics();
    
    /**
     * Update service configuration
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
     * Get data storage rate
     * 
     * @return Data stored per second
     */
    double getDataStorageRate();
    
    /**
     * Get error count
     * 
     * @return Number of errors encountered
     */
    int getErrorCount();
    
    /**
     * Reset service error count
     */
    void resetErrorCount();
    
    /**
     * Get last task processing timestamp
     * 
     * @return Timestamp of last task processing
     */
    long getLastTaskProcessingTime();
    
    /**
     * Get last data storage timestamp
     * 
     * @return Timestamp of last data storage
     */
    long getLastDataStorageTime();
} 