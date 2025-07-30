package com.nci.fogedge.cloud;

import com.nci.fogedge.network.NetworkManager;
import com.nci.fogedge.utils.MetricsCollector;

/**
 * Base interface for all Cloud Services in the Fog and Edge Computing System
 * 
 * This interface defines the contract that all cloud computing services must implement,
 * including data analytics, machine learning, storage, and orchestration services.
 * It provides methods for service lifecycle management and task processing.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public interface CloudService {
    
    /**
     * Get the unique identifier for this cloud service
     * 
     * @return Service ID string
     */
    String getServiceId();
    
    /**
     * Get the type of this cloud service
     * 
     * @return Service type (e.g., "DATA_ANALYTICS", "MACHINE_LEARNING", "STORAGE")
     */
    String getServiceType();
    
    /**
     * Get the current status of the service
     * 
     * @return Service status (e.g., "ACTIVE", "INACTIVE", "OVERLOADED", "ERROR")
     */
    String getStatus();
    
    /**
     * Check if the service is healthy and functioning properly
     * 
     * @return true if service is healthy, false otherwise
     */
    boolean isHealthy();
    
    /**
     * Start the cloud service and begin task processing
     */
    void start();
    
    /**
     * Stop the cloud service and cease all operations
     */
    void stop();
    
    /**
     * Process a task received from edge nodes
     * 
     * @param task Task to process
     * @return Processing result
     */
    Object processTask(Object task);
    
    /**
     * Get the current CPU utilization of the service
     * 
     * @return CPU utilization as percentage (0-100)
     */
    double getCpuUtilization();
    
    /**
     * Get the current memory utilization of the service
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
     * Get the total tasks processed by this service
     * 
     * @return Total number of tasks processed
     */
    int getTotalTasksProcessed();
    
    /**
     * Get the average processing time
     * 
     * @return Average processing time in milliseconds
     */
    double getAverageProcessingTime();
    
    /**
     * Get the service efficiency
     * 
     * @return Service efficiency as percentage (0-100)
     */
    double getServiceEfficiency();
    
    /**
     * Get service configuration parameters
     * 
     * @return Map of configuration parameters
     */
    java.util.Map<String, Object> getConfiguration();
    
    /**
     * Update service configuration
     * 
     * @param config New configuration parameters
     */
    void updateConfiguration(java.util.Map<String, Object> config);
    
    /**
     * Get service performance metrics
     * 
     * @return Map containing performance metrics
     */
    java.util.Map<String, Object> getPerformanceMetrics();
    
    /**
     * Reset service statistics and counters
     */
    void resetStatistics();
    
    /**
     * Perform service self-diagnostic
     * 
     * @return Diagnostic result
     */
    DiagnosticResult performDiagnostic();
    
    /**
     * Check if the service can handle additional tasks
     * 
     * @return true if service can accept more tasks, false otherwise
     */
    boolean canAcceptTasks();
    
    /**
     * Get the current queue length
     * 
     * @return Number of tasks in the processing queue
     */
    int getQueueLength();
    
    /**
     * Result of service diagnostic operation
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