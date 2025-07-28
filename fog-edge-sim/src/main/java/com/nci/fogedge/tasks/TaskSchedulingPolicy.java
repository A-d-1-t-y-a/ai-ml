package com.nci.fogedge.tasks;

/**
 * Enum representing different policies for scheduling tasks in the simulation.
 */
public enum TaskSchedulingPolicy {
    /**
     * First-In-First-Out: Tasks are scheduled in the order they are submitted
     */
    FIFO,
    
    /**
     * Priority-based: Tasks with higher priority are scheduled first
     */
    PRIORITY,
    
    /**
     * Shortest Job First: Tasks with shorter duration are scheduled first
     */
    SHORTEST_JOB_FIRST,
    
    /**
     * Local Only: Tasks are executed only on the source device
     */
    LOCAL_ONLY,
    
    /**
     * Edge First: Prefer edge nodes over fog nodes and cloud datacenters
     */
    EDGE_FIRST,
    
    /**
     * Fog First: Prefer fog nodes over edge nodes and cloud datacenters
     */
    FOG_FIRST,
    
    /**
     * Cloud First: Prefer cloud datacenters over fog nodes and edge nodes
     */
    CLOUD_FIRST,
    
    /**
     * Resource Aware: Schedule tasks based on available resources
     */
    RESOURCE_AWARE,
    
    /**
     * Security Aware: Schedule tasks based on security considerations
     */
    SECURITY_AWARE,
    
    /**
     * Energy Aware: Schedule tasks based on energy efficiency
     */
    ENERGY_AWARE,
    
    /**
     * Cost Aware: Schedule tasks based on execution cost
     */
    COST_AWARE,
    
    /**
     * Latency Aware: Schedule tasks based on network latency
     */
    LATENCY_AWARE,
    
    /**
     * Security First: Security-critical tasks are scheduled first
     */
    SECURITY_FIRST
}
