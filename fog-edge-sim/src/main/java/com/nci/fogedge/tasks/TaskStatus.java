package com.nci.fogedge.tasks;

/**
 * Enum representing the different states a task can be in during its lifecycle.
 */
public enum TaskStatus {
    /**
     * Task has been created but not yet submitted for execution
     */
    CREATED,
    
    /**
     * Task has been submitted for execution but not yet started
     */
    SUBMITTED,
    
    /**
     * Task is currently being executed
     */
    EXECUTING,
    
    /**
     * Task has completed execution successfully
     */
    COMPLETED,
    
    /**
     * Task execution has failed
     */
    FAILED,
    
    /**
     * Task has been cancelled before completion
     */
    CANCELLED
}
