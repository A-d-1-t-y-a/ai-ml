package com.nci.fogedge.edge;

import com.nci.fogedge.network.NetworkManager;
import com.nci.fogedge.utils.MetricsCollector;

/**
 * Base interface for all Edge Nodes in the Fog and Edge Computing System
 * 
 * This interface defines the contract that all edge computing nodes must implement,
 * including data processing, analytics, and gateway nodes. It provides methods for
 * node lifecycle management, data processing, and task offloading decisions.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public interface EdgeNode {
    
    /**
     * Get the unique identifier for this edge node
     * 
     * @return Node ID string
     */
    String getNodeId();
    
    /**
     * Get the type of this edge node
     * 
     * @return Node type (e.g., "DATA_PROCESSING", "ANALYTICS", "GATEWAY")
     */
    String getNodeType();
    
    /**
     * Get the current status of the node
     * 
     * @return Node status (e.g., "ACTIVE", "INACTIVE", "OVERLOADED", "ERROR")
     */
    String getStatus();
    
    /**
     * Check if the node is healthy and functioning properly
     * 
     * @return true if node is healthy, false otherwise
     */
    boolean isHealthy();
    
    /**
     * Start the edge node and begin data processing
     */
    void start();
    
    /**
     * Stop the edge node and cease all operations
     */
    void stop();
    
    /**
     * Process incoming data from IoT devices
     * 
     * @param data Data to process
     * @return Processing result
     */
    Object processData(Object data);
    
    /**
     * Get the current CPU utilization of the node
     * 
     * @return CPU utilization as percentage (0-100)
     */
    double getCpuUtilization();
    
    /**
     * Get the current memory utilization of the node
     * 
     * @return Memory utilization as percentage (0-100)
     */
    double getMemoryUtilization();
    
    /**
     * Get the current network bandwidth utilization
     * 
     * @return Bandwidth utilization as percentage (0-100)
     */
    double getBandwidthUtilization();
    
    /**
     * Get the total data processed by this node
     * 
     * @return Total data in bytes
     */
    long getTotalDataProcessed();
    
    /**
     * Get the number of tasks offloaded to cloud
     * 
     * @return Count of offloaded tasks
     */
    int getTasksOffloaded();
    
    /**
     * Get the average processing time
     * 
     * @return Average processing time in milliseconds
     */
    double getAverageProcessingTime();
    
    /**
     * Get the task offloading rate
     * 
     * @return Offloading rate as percentage (0-100)
     */
    double getOffloadingRate();
    
    /**
     * Get node configuration parameters
     * 
     * @return Map of configuration parameters
     */
    java.util.Map<String, Object> getConfiguration();
    
    /**
     * Update node configuration
     * 
     * @param config New configuration parameters
     */
    void updateConfiguration(java.util.Map<String, Object> config);
    
    /**
     * Get node performance metrics
     * 
     * @return Map containing performance metrics
     */
    java.util.Map<String, Object> getPerformanceMetrics();
    
    /**
     * Reset node statistics and counters
     */
    void resetStatistics();
    
    /**
     * Perform node self-diagnostic
     * 
     * @return Diagnostic result
     */
    DiagnosticResult performDiagnostic();
    
    /**
     * Decide whether to offload a task to the cloud
     * 
     * @param task Task to evaluate for offloading
     * @return true if task should be offloaded, false otherwise
     */
    boolean shouldOffloadTask(Object task);
    
    /**
     * Offload a task to the cloud layer
     * 
     * @param task Task to offload
     * @return true if offloading successful, false otherwise
     */
    boolean offloadTask(Object task);
    
    /**
     * Result of node diagnostic operation
     */
    class DiagnosticResult {
        private final boolean passed;
        private final String message;
        private final java.util.Map<String, Object> details;
        
        public DiagnosticResult(boolean passed, String message, java.util.Map<String, Object> details) {
            this.passed = passed;
            this.message = message;
            this.details = details;
        }
        
        public boolean isPassed() {
            return passed;
        }
        
        public String getMessage() {
            return message;
        }
        
        public java.util.Map<String, Object> getDetails() {
            return details;
        }
    }
} 