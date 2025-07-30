package com.nci.fogedge.utils;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/**
 * Device Metrics for Fog and Edge Computing System
 * 
 * This class tracks performance metrics for IoT devices in the system.
 * It maintains historical data and calculates aggregate statistics.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class DeviceMetrics {
    
    private final String deviceId;
    private final List<PerformanceMetrics> historicalMetrics;
    private PerformanceMetrics currentMetrics;
    private Instant lastUpdate;
    
    /**
     * Constructor for DeviceMetrics
     * 
     * @param deviceId Device identifier
     */
    public DeviceMetrics(String deviceId) {
        this.deviceId = deviceId;
        this.historicalMetrics = new ArrayList<>();
        this.lastUpdate = Instant.now();
    }
    
    /**
     * Get device identifier
     * 
     * @return Device ID
     */
    public String getDeviceId() {
        return deviceId;
    }
    
    /**
     * Update device metrics
     * 
     * @param metrics New performance metrics
     */
    public void updateMetrics(PerformanceMetrics metrics) {
        this.currentMetrics = metrics;
        this.historicalMetrics.add(metrics);
        this.lastUpdate = Instant.now();
        
        // Keep only last 100 metrics to prevent memory issues
        if (historicalMetrics.size() > 100) {
            historicalMetrics.remove(0);
        }
    }
    
    /**
     * Get current metrics
     * 
     * @return Current performance metrics
     */
    public PerformanceMetrics getCurrentMetrics() {
        return currentMetrics;
    }
    
    /**
     * Get historical metrics
     * 
     * @return List of historical metrics
     */
    public List<PerformanceMetrics> getHistoricalMetrics() {
        return new ArrayList<>(historicalMetrics);
    }
    
    /**
     * Get last update time
     * 
     * @return Last update timestamp
     */
    public Instant getLastUpdate() {
        return lastUpdate;
    }
    
    /**
     * Get average latency
     * 
     * @return Average latency in milliseconds
     */
    public double getAverageLatency() {
        if (historicalMetrics.isEmpty()) {
            return 0.0;
        }
        
        return historicalMetrics.stream()
            .mapToDouble(PerformanceMetrics::getLatency)
            .average()
            .orElse(0.0);
    }
    
    /**
     * Get average throughput
     * 
     * @return Average throughput in Mbps
     */
    public double getAverageThroughput() {
        if (historicalMetrics.isEmpty()) {
            return 0.0;
        }
        
        return historicalMetrics.stream()
            .mapToDouble(PerformanceMetrics::getThroughput)
            .average()
            .orElse(0.0);
    }
    
    /**
     * Get average energy consumption
     * 
     * @return Average energy consumption in watts
     */
    public double getAverageEnergyConsumption() {
        if (historicalMetrics.isEmpty()) {
            return 0.0;
        }
        
        return historicalMetrics.stream()
            .mapToDouble(PerformanceMetrics::getEnergyConsumption)
            .average()
            .orElse(0.0);
    }
    
    /**
     * Get total data processed
     * 
     * @return Total data processed in bytes
     */
    public long getTotalDataProcessed() {
        return historicalMetrics.stream()
            .mapToLong(PerformanceMetrics::getDataProcessed)
            .sum();
    }
    
    /**
     * Get health score
     * 
     * @return Health score (0-100)
     */
    public double getHealthScore() {
        if (currentMetrics == null) {
            return 50.0; // Default score
        }
        
        double latencyScore = Math.max(0.0, 100.0 - currentMetrics.getLatency() / 2.0);
        double throughputScore = Math.min(100.0, currentMetrics.getThroughput() / 2.0);
        double energyScore = Math.max(0.0, 100.0 - currentMetrics.getEnergyConsumption() / 2.0);
        double healthScore = currentMetrics.getHealthScore();
        
        return (latencyScore + throughputScore + energyScore + healthScore) / 4.0;
    }
    
    /**
     * Get performance trend
     * 
     * @return Performance trend (IMPROVING, STABLE, DEGRADING)
     */
    public PerformanceTrend getPerformanceTrend() {
        if (historicalMetrics.size() < 5) {
            return PerformanceTrend.STABLE;
        }
        
        // Compare recent metrics with older ones
        int recentCount = Math.min(5, historicalMetrics.size() / 2);
        int olderCount = Math.min(5, historicalMetrics.size() - recentCount);
        
        double recentAvg = historicalMetrics.stream()
            .skip(historicalMetrics.size() - recentCount)
            .mapToDouble(PerformanceMetrics::getPerformanceScore)
            .average()
            .orElse(0.0);
        
        double olderAvg = historicalMetrics.stream()
            .limit(olderCount)
            .mapToDouble(PerformanceMetrics::getPerformanceScore)
            .average()
            .orElse(0.0);
        
        double difference = recentAvg - olderAvg;
        
        if (difference > 10.0) {
            return PerformanceTrend.IMPROVING;
        } else if (difference < -10.0) {
            return PerformanceTrend.DEGRADING;
        } else {
            return PerformanceTrend.STABLE;
        }
    }
    
    /**
     * Check if device is healthy
     * 
     * @return True if device is healthy
     */
    public boolean isHealthy() {
        return getHealthScore() > 70.0 && currentMetrics != null && currentMetrics.isPerformanceGood();
    }
    
    /**
     * Get uptime percentage
     * 
     * @return Uptime percentage (0-100)
     */
    public double getUptimePercentage() {
        if (historicalMetrics.isEmpty()) {
            return 0.0;
        }
        
        long healthyCount = historicalMetrics.stream()
            .mapToDouble(PerformanceMetrics::getHealthScore)
            .filter(score -> score > 70.0)
            .count();
        
        return (double) healthyCount / historicalMetrics.size() * 100.0;
    }
    
    /**
     * Performance trend enumeration
     */
    public enum PerformanceTrend {
        IMPROVING,
        STABLE,
        DEGRADING
    }
    
    @Override
    public String toString() {
        return String.format("DeviceMetrics{device=%s, health=%.1f, trend=%s, data=%d bytes}",
            deviceId, getHealthScore(), getPerformanceTrend(), getTotalDataProcessed());
    }
} 