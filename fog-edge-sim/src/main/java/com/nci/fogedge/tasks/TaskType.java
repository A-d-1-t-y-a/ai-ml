package com.nci.fogedge.tasks;

/**
 * Enum representing the different types of tasks that can be executed in the simulation.
 */
public enum TaskType {
    /**
     * General computational task
     * Requires CPU resources
     */
    COMPUTATIONAL,
    
    /**
     * Data processing task
     * Requires CPU and memory resources
     */
    DATA_PROCESSING,
    
    /**
     * Storage task
     * Requires storage resources
     */
    STORAGE,
    
    /**
     * Machine learning inference task
     * Requires CPU/GPU resources and memory
     */
    ML_INFERENCE,
    
    /**
     * Machine learning training task
     * Requires significant CPU/GPU resources, memory, and storage
     */
    ML_TRAINING,
    
    /**
     * Real-time task with strict latency requirements
     * Must be completed within a specific time frame
     */
    REAL_TIME,
    
    /**
     * Batch processing task
     * Can be delayed and processed in batches
     */
    BATCH,
    
    /**
     * Security-related task
     * Includes encryption, decryption, authentication, etc.
     */
    SECURITY
}
