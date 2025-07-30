package com.nci.fogedge.utils;

import com.nci.fogedge.network.NetworkStatistics;

import java.util.Map;

/**
 * System Metrics for Fog and Edge Computing System
 * 
 * This class holds comprehensive system-wide metrics for the entire fog and edge computing system.
 * It aggregates metrics from IoT devices, edge nodes, cloud services, and network components.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class SystemMetrics {
    
    private final long totalDataProcessed;
    private final long totalDevicesActive;
    private final long totalNodesActive;
    private final double averageLatency;
    private final double latencyReduction;
    private final double dataReductionAtEdge;
    private final double energyEfficiency;
    private final double bandwidthOptimization;
    private final NetworkStatistics networkStatistics;
    private final Map<String, DeviceMetrics> deviceMetrics;
    private final Map<String, NodeMetrics> nodeMetrics;
    
    /**
     * Constructor for SystemMetrics
     * 
     * @param totalDataProcessed Total data processed in bytes
     * @param totalDevicesActive Total active devices
     * @param totalNodesActive Total active nodes
     * @param averageLatency Average latency in milliseconds
     * @param latencyReduction Latency reduction percentage
     * @param dataReductionAtEdge Data reduction at edge percentage
     * @param energyEfficiency Energy efficiency percentage
     * @param bandwidthOptimization Bandwidth optimization percentage
     * @param networkStatistics Network statistics
     * @param deviceMetrics Device-specific metrics
     * @param nodeMetrics Node-specific metrics
     */
    public SystemMetrics(long totalDataProcessed, long totalDevicesActive, long totalNodesActive,
                        double averageLatency, double latencyReduction, double dataReductionAtEdge,
                        double energyEfficiency, double bandwidthOptimization,
                        NetworkStatistics networkStatistics,
                        Map<String, DeviceMetrics> deviceMetrics,
                        Map<String, NodeMetrics> nodeMetrics) {
        this.totalDataProcessed = totalDataProcessed;
        this.totalDevicesActive = totalDevicesActive;
        this.totalNodesActive = totalNodesActive;
        this.averageLatency = averageLatency;
        this.latencyReduction = latencyReduction;
        this.dataReductionAtEdge = dataReductionAtEdge;
        this.energyEfficiency = energyEfficiency;
        this.bandwidthOptimization = bandwidthOptimization;
        this.networkStatistics = networkStatistics;
        this.deviceMetrics = deviceMetrics;
        this.nodeMetrics = nodeMetrics;
    }
    
    /**
     * Get total data processed
     * 
     * @return Total data processed in bytes
     */
    public long getTotalDataProcessed() {
        return totalDataProcessed;
    }
    
    /**
     * Get total active devices
     * 
     * @return Total active devices
     */
    public long getTotalDevicesActive() {
        return totalDevicesActive;
    }
    
    /**
     * Get total active nodes
     * 
     * @return Total active nodes
     */
    public long getTotalNodesActive() {
        return totalNodesActive;
    }
    
    /**
     * Get average latency
     * 
     * @return Average latency in milliseconds
     */
    public double getAverageLatency() {
        return averageLatency;
    }
    
    /**
     * Get latency reduction
     * 
     * @return Latency reduction percentage
     */
    public double getLatencyReduction() {
        return latencyReduction;
    }
    
    /**
     * Get data reduction at edge
     * 
     * @return Data reduction at edge percentage
     */
    public double getDataReductionAtEdge() {
        return dataReductionAtEdge;
    }
    
    /**
     * Get energy efficiency
     * 
     * @return Energy efficiency percentage
     */
    public double getEnergyEfficiency() {
        return energyEfficiency;
    }
    
    /**
     * Get bandwidth optimization
     * 
     * @return Bandwidth optimization percentage
     */
    public double getBandwidthOptimization() {
        return bandwidthOptimization;
    }
    
    /**
     * Get network statistics
     * 
     * @return Network statistics
     */
    public NetworkStatistics getNetworkStatistics() {
        return networkStatistics;
    }
    
    /**
     * Get device metrics
     * 
     * @return Device-specific metrics
     */
    public Map<String, DeviceMetrics> getDeviceMetrics() {
        return deviceMetrics;
    }
    
    /**
     * Get node metrics
     * 
     * @return Node-specific metrics
     */
    public Map<String, NodeMetrics> getNodeMetrics() {
        return nodeMetrics;
    }
    
    /**
     * Get overall system health score
     * 
     * @return System health score (0-100)
     */
    public double getSystemHealthScore() {
        double deviceHealth = calculateDeviceHealth();
        double nodeHealth = calculateNodeHealth();
        double networkHealth = networkStatistics != null ? networkStatistics.getNetworkHealthScore() : 50.0;
        double performanceHealth = calculatePerformanceHealth();
        
        return (deviceHealth + nodeHealth + networkHealth + performanceHealth) / 4.0;
    }
    
    /**
     * Calculate device health score
     * 
     * @return Device health score
     */
    private double calculateDeviceHealth() {
        if (deviceMetrics.isEmpty()) {
            return 50.0; // Default score
        }
        
        double totalHealth = 0.0;
        int deviceCount = 0;
        
        for (DeviceMetrics metrics : deviceMetrics.values()) {
            totalHealth += metrics.getHealthScore();
            deviceCount++;
        }
        
        return deviceCount > 0 ? totalHealth / deviceCount : 50.0;
    }
    
    /**
     * Calculate node health score
     * 
     * @return Node health score
     */
    private double calculateNodeHealth() {
        if (nodeMetrics.isEmpty()) {
            return 50.0; // Default score
        }
        
        double totalHealth = 0.0;
        int nodeCount = 0;
        
        for (NodeMetrics metrics : nodeMetrics.values()) {
            totalHealth += metrics.getHealthScore();
            nodeCount++;
        }
        
        return nodeCount > 0 ? totalHealth / nodeCount : 50.0;
    }
    
    /**
     * Calculate performance health score
     * 
     * @return Performance health score
     */
    private double calculatePerformanceHealth() {
        double latencyScore = Math.max(0.0, 100.0 - averageLatency / 2.0);
        double reductionScore = (latencyReduction + dataReductionAtEdge + energyEfficiency + bandwidthOptimization) / 4.0;
        
        return (latencyScore + reductionScore) / 2.0;
    }
    
    /**
     * Check if system is healthy
     * 
     * @return True if system is healthy
     */
    public boolean isSystemHealthy() {
        return getSystemHealthScore() > 70.0 &&
               averageLatency < 100.0 &&
               latencyReduction > 20.0 &&
               dataReductionAtEdge > 30.0;
    }
    
    /**
     * Get system efficiency score
     * 
     * @return System efficiency score (0-100)
     */
    public double getSystemEfficiencyScore() {
        return (latencyReduction + dataReductionAtEdge + energyEfficiency + bandwidthOptimization) / 4.0;
    }
    
    /**
     * Get total active entities
     * 
     * @return Total active entities (devices + nodes)
     */
    public long getTotalActiveEntities() {
        return totalDevicesActive + totalNodesActive;
    }
    
    /**
     * Get data processing rate
     * 
     * @return Data processing rate in bytes per second
     */
    public double getDataProcessingRate() {
        // Simplified calculation - in real system, this would be time-based
        return totalDataProcessed / 3600.0; // Assume 1 hour of operation
    }
    
    @Override
    public String toString() {
        return String.format("SystemMetrics{data=%d bytes, devices=%d, nodes=%d, latency=%.2fms, health=%.1f}",
            totalDataProcessed, totalDevicesActive, totalNodesActive, averageLatency, getSystemHealthScore());
    }
} 