package com.nci.fogedge.utils;

import java.util.Map;
import java.util.HashMap;

/**
 * Performance Metrics for Fog and Edge Computing System
 * 
 * This class represents performance metrics for various system components
 * including IoT devices, edge nodes, and cloud services.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class PerformanceMetrics {
    
    private final String componentId;
    private final String componentType;
    private final long timestamp;
    private final Map<String, Object> metrics;
    
    /**
     * Constructor for PerformanceMetrics
     * 
     * @param componentId Component identifier
     * @param componentType Component type
     */
    public PerformanceMetrics(String componentId, String componentType) {
        this.componentId = componentId;
        this.componentType = componentType;
        this.timestamp = System.currentTimeMillis();
        this.metrics = new HashMap<>();
    }
    
    /**
     * Get component ID
     * 
     * @return Component identifier
     */
    public String getComponentId() {
        return componentId;
    }
    
    /**
     * Get component type
     * 
     * @return Component type
     */
    public String getComponentType() {
        return componentType;
    }
    
    /**
     * Get timestamp
     * 
     * @return Timestamp when metrics were collected
     */
    public long getTimestamp() {
        return timestamp;
    }
    
    /**
     * Add a metric
     * 
     * @param key Metric key
     * @param value Metric value
     */
    public void addMetric(String key, Object value) {
        metrics.put(key, value);
    }
    
    /**
     * Get a metric value
     * 
     * @param key Metric key
     * @return Metric value
     */
    public Object getMetric(String key) {
        return metrics.get(key);
    }
    
    /**
     * Get all metrics
     * 
     * @return Map of all metrics
     */
    public Map<String, Object> getAllMetrics() {
        return new HashMap<>(metrics);
    }
    
    /**
     * Get CPU usage
     * 
     * @return CPU usage percentage
     */
    public double getCpuUsage() {
        Object value = metrics.get("cpu_usage");
        return value instanceof Number ? ((Number) value).doubleValue() : 0.0;
    }
    
    /**
     * Get memory usage
     * 
     * @return Memory usage percentage
     */
    public double getMemoryUsage() {
        Object value = metrics.get("memory_usage");
        return value instanceof Number ? ((Number) value).doubleValue() : 0.0;
    }
    
    /**
     * Get storage usage
     * 
     * @return Storage usage percentage
     */
    public double getStorageUsage() {
        Object value = metrics.get("storage_usage");
        return value instanceof Number ? ((Number) value).doubleValue() : 0.0;
    }
    
    /**
     * Get bandwidth usage
     * 
     * @return Bandwidth usage in Mbps
     */
    public double getBandwidthUsage() {
        Object value = metrics.get("bandwidth_usage");
        return value instanceof Number ? ((Number) value).doubleValue() : 0.0;
    }
    
    /**
     * Get energy consumption
     * 
     * @return Energy consumption in watts
     */
    public double getEnergyConsumption() {
        Object value = metrics.get("energy_consumption");
        return value instanceof Number ? ((Number) value).doubleValue() : 0.0;
    }
    
    /**
     * Get latency
     * 
     * @return Latency in milliseconds
     */
    public double getLatency() {
        Object value = metrics.get("latency");
        return value instanceof Number ? ((Number) value).doubleValue() : 0.0;
    }
    
    /**
     * Get throughput
     * 
     * @return Throughput in Mbps
     */
    public double getThroughput() {
        Object value = metrics.get("throughput");
        return value instanceof Number ? ((Number) value).doubleValue() : 0.0;
    }
    
    /**
     * Get error count
     * 
     * @return Number of errors
     */
    public int getErrorCount() {
        Object value = metrics.get("error_count");
        return value instanceof Number ? ((Number) value).intValue() : 0;
    }
    
    /**
     * Get success rate
     * 
     * @return Success rate percentage
     */
    public double getSuccessRate() {
        Object value = metrics.get("success_rate");
        return value instanceof Number ? ((Number) value).doubleValue() : 0.0;
    }
    
    /**
     * Get data processed
     * 
     * @return Data processed in bytes
     */
    public long getDataProcessed() {
        Object value = metrics.get("data_processed");
        return value instanceof Number ? ((Number) value).longValue() : 0L;
    }
    
    /**
     * Get tasks processed
     * 
     * @return Number of tasks processed
     */
    public int getTasksProcessed() {
        Object value = metrics.get("tasks_processed");
        return value instanceof Number ? ((Number) value).intValue() : 0;
    }
    
    /**
     * Get battery level
     * 
     * @return Battery level percentage
     */
    public double getBatteryLevel() {
        Object value = metrics.get("battery_level");
        return value instanceof Number ? ((Number) value).doubleValue() : 0.0;
    }
    
    /**
     * Get signal strength
     * 
     * @return Signal strength in dBm
     */
    public double getSignalStrength() {
        Object value = metrics.get("signal_strength");
        return value instanceof Number ? ((Number) value).doubleValue() : 0.0;
    }
    
    /**
     * Check if component is healthy
     * 
     * @return True if component is healthy
     */
    public boolean isHealthy() {
        Object value = metrics.get("healthy");
        return value instanceof Boolean ? (Boolean) value : false;
    }
    
    /**
     * Check if component is running
     * 
     * @return True if component is running
     */
    public boolean isRunning() {
        Object value = metrics.get("running");
        return value instanceof Boolean ? (Boolean) value : false;
    }
    
    /**
     * Get health score
     * 
     * @return Health score (0-100)
     */
    public double getHealthScore() {
        Object value = metrics.get("health_score");
        return value instanceof Number ? ((Number) value).doubleValue() : 100.0;
    }
    
    /**
     * Get performance score
     * 
     * @return Performance score (0-100)
     */
    public double getPerformanceScore() {
        // Calculate performance score based on metrics
        double score = 100.0;
        
        // Deduct points for high latency
        double latency = getLatency();
        if (latency > 200.0) {
            score -= 30.0;
        } else if (latency > 100.0) {
            score -= 15.0;
        }
        
        // Deduct points for low throughput
        double throughput = getThroughput();
        if (throughput < 10.0) {
            score -= 25.0;
        } else if (throughput < 50.0) {
            score -= 10.0;
        }
        
        // Deduct points for high resource usage
        double cpuUsage = getCpuUsage();
        if (cpuUsage > 90.0) {
            score -= 20.0;
        } else if (cpuUsage > 80.0) {
            score -= 10.0;
        }
        
        double memoryUsage = getMemoryUsage();
        if (memoryUsage > 90.0) {
            score -= 20.0;
        } else if (memoryUsage > 80.0) {
            score -= 10.0;
        }
        
        // Deduct points for errors
        int errorCount = getErrorCount();
        score -= errorCount * 2.0;
        
        return Math.max(0.0, score);
    }
    
    /**
     * Check if performance is good
     * 
     * @return True if performance is good
     */
    public boolean isPerformanceGood() {
        return getPerformanceScore() >= 70.0 && 
               getLatency() < 100.0 && 
               getThroughput() > 10.0 && 
               getCpuUsage() < 80.0 && 
               getMemoryUsage() < 80.0;
    }
    
    @Override
    public String toString() {
        return String.format("PerformanceMetrics{componentId='%s', componentType='%s', timestamp=%d, metrics=%s}",
            componentId, componentType, timestamp, metrics);
    }
} 