package com.nci.fogedge.utils;

import java.time.Instant;

/**
 * Performance Metrics for Fog and Edge Computing System
 * 
 * This class represents performance metrics for IoT devices and edge nodes.
 * It tracks latency, throughput, energy consumption, and other key performance indicators.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class PerformanceMetrics {
    
    private final String entityId;
    private final String entityType;
    private final Instant timestamp;
    private final double latency;
    private final double throughput;
    private final double energyConsumption;
    private final double cpuUsage;
    private final double memoryUsage;
    private final long dataProcessed;
    private final int activeConnections;
    private final double healthScore;
    
    /**
     * Constructor for PerformanceMetrics
     * 
     * @param entityId Entity identifier
     * @param entityType Entity type (DEVICE, NODE, SERVICE)
     * @param latency Latency in milliseconds
     * @param throughput Throughput in Mbps
     * @param energyConsumption Energy consumption in watts
     * @param cpuUsage CPU usage percentage
     * @param memoryUsage Memory usage percentage
     * @param dataProcessed Data processed in bytes
     * @param activeConnections Number of active connections
     * @param healthScore Health score (0-100)
     */
    public PerformanceMetrics(String entityId, String entityType, double latency, double throughput,
                            double energyConsumption, double cpuUsage, double memoryUsage,
                            long dataProcessed, int activeConnections, double healthScore) {
        this.entityId = entityId;
        this.entityType = entityType;
        this.timestamp = Instant.now();
        this.latency = latency;
        this.throughput = throughput;
        this.energyConsumption = energyConsumption;
        this.cpuUsage = cpuUsage;
        this.memoryUsage = memoryUsage;
        this.dataProcessed = dataProcessed;
        this.activeConnections = activeConnections;
        this.healthScore = healthScore;
    }
    
    /**
     * Get entity identifier
     * 
     * @return Entity ID
     */
    public String getEntityId() {
        return entityId;
    }
    
    /**
     * Get entity type
     * 
     * @return Entity type
     */
    public String getEntityType() {
        return entityType;
    }
    
    /**
     * Get timestamp
     * 
     * @return Timestamp
     */
    public Instant getTimestamp() {
        return timestamp;
    }
    
    /**
     * Get latency
     * 
     * @return Latency in milliseconds
     */
    public double getLatency() {
        return latency;
    }
    
    /**
     * Get throughput
     * 
     * @return Throughput in Mbps
     */
    public double getThroughput() {
        return throughput;
    }
    
    /**
     * Get energy consumption
     * 
     * @return Energy consumption in watts
     */
    public double getEnergyConsumption() {
        return energyConsumption;
    }
    
    /**
     * Get CPU usage
     * 
     * @return CPU usage percentage
     */
    public double getCpuUsage() {
        return cpuUsage;
    }
    
    /**
     * Get memory usage
     * 
     * @return Memory usage percentage
     */
    public double getMemoryUsage() {
        return memoryUsage;
    }
    
    /**
     * Get data processed
     * 
     * @return Data processed in bytes
     */
    public long getDataProcessed() {
        return dataProcessed;
    }
    
    /**
     * Get active connections
     * 
     * @return Number of active connections
     */
    public int getActiveConnections() {
        return activeConnections;
    }
    
    /**
     * Get health score
     * 
     * @return Health score (0-100)
     */
    public double getHealthScore() {
        return healthScore;
    }
    
    /**
     * Check if performance is good
     * 
     * @return True if performance is good
     */
    public boolean isPerformanceGood() {
        return latency < 100.0 && // Less than 100ms latency
               throughput > 10.0 && // More than 10 Mbps throughput
               cpuUsage < 80.0 && // Less than 80% CPU usage
               memoryUsage < 80.0 && // Less than 80% memory usage
               healthScore > 70.0; // Health score above 70%
    }
    
    /**
     * Get performance score
     * 
     * @return Performance score (0-100)
     */
    public double getPerformanceScore() {
        double latencyScore = Math.max(0.0, 100.0 - latency / 2.0); // Normalize latency
        double throughputScore = Math.min(100.0, throughput / 2.0); // Normalize throughput
        double resourceScore = (100.0 - cpuUsage + 100.0 - memoryUsage) / 2.0; // Resource efficiency
        double healthScore = this.healthScore;
        
        return (latencyScore + throughputScore + resourceScore + healthScore) / 4.0;
    }
    
    /**
     * Get energy efficiency
     * 
     * @return Energy efficiency score (0-100)
     */
    public double getEnergyEfficiency() {
        // Calculate energy efficiency based on throughput vs energy consumption
        if (energyConsumption > 0) {
            double efficiency = (throughput / energyConsumption) * 10.0; // Normalize
            return Math.min(100.0, Math.max(0.0, efficiency));
        }
        return 0.0;
    }
    
    @Override
    public String toString() {
        return String.format("PerformanceMetrics{entity=%s, type=%s, latency=%.2fms, throughput=%.2fMbps, health=%.1f}",
            entityId, entityType, latency, throughput, healthScore);
    }
} 