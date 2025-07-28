package com.nci.fogedge.tasks;

/**
 * Enum representing the different states a task can be in during its lifecycle.
 */
public enum TaskStatus {
    /**
     * Task has been created but is not yet ready for execution
     */
    CREATED,
    
    /**
     * Task is ready for execution
     */
    READY,
    
    /**
     * Task is currently being executed
     */
    RUNNING,
    
    /**
     * Task has been completed successfully
     */
    COMPLETED,
    
    /**
     * Task has failed to complete
     */
    FAILED,
    
    /**
     * Task has been offloaded to another device
     */
    OFFLOADED,
    
    /**
     * Task has been cancelled
     */
    CANCELLED,
    
    /**
     * Task is waiting for resources
     */
    WAITING
}
