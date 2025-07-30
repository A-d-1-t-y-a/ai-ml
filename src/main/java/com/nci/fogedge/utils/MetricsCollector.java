package com.nci.fogedge.utils;

import com.nci.fogedge.network.NetworkStatistics;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Metrics Collector for Fog and Edge Computing System
 * 
 * This class collects and manages performance metrics for the entire system.
 * It tracks metrics for IoT devices, edge nodes, cloud services, and network performance.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class MetricsCollector {
    
    private static final Logger logger = LoggerFactory.getLogger(MetricsCollector.class);
    
    // System-wide metrics
    private final AtomicLong totalDataProcessed;
    private final AtomicLong totalDevicesActive;
    private final AtomicLong totalNodesActive;
    private final AtomicReference<Double> averageLatency;
    private final AtomicReference<Double> latencyReduction;
    private final AtomicReference<Double> dataReductionAtEdge;
    private final AtomicReference<Double> energyEfficiency;
    private final AtomicReference<Double> bandwidthOptimization;
    
    // Device-specific metrics
    private final Map<String, DeviceMetrics> deviceMetrics;
    
    // Node-specific metrics
    private final Map<String, NodeMetrics> nodeMetrics;
    
    // Network metrics
    private NetworkStatistics networkStatistics;
    
    // Performance tracking
    private final Map<String, PerformanceMetrics> performanceMetrics;
    
    /**
     * Constructor for MetricsCollector
     */
    public MetricsCollector() {
        this.totalDataProcessed = new AtomicLong(0);
        this.totalDevicesActive = new AtomicLong(0);
        this.totalNodesActive = new AtomicLong(0);
        this.averageLatency = new AtomicReference<>(0.0);
        this.latencyReduction = new AtomicReference<>(0.0);
        this.dataReductionAtEdge = new AtomicReference<>(0.0);
        this.energyEfficiency = new AtomicReference<>(0.0);
        this.bandwidthOptimization = new AtomicReference<>(0.0);
        
        this.deviceMetrics = new ConcurrentHashMap<>();
        this.nodeMetrics = new ConcurrentHashMap<>();
        this.performanceMetrics = new ConcurrentHashMap<>();
        
        logger.info("MetricsCollector initialized");
    }
    
    /**
     * Collect system-wide metrics
     */
    public void collectMetrics() {
        try {
            // Calculate aggregate metrics
            calculateAggregateMetrics();
            
            // Update performance indicators
            updatePerformanceIndicators();
            
            logger.debug("Metrics collection completed");
            
        } catch (Exception e) {
            logger.error("Error during metrics collection", e);
        }
    }
    
    /**
     * Update device metrics
     * 
     * @param deviceId Device identifier
     * @param metrics Device performance metrics
     */
    public void updateDeviceMetrics(String deviceId, PerformanceMetrics metrics) {
        DeviceMetrics deviceMetrics = this.deviceMetrics.computeIfAbsent(deviceId, 
            k -> new DeviceMetrics(deviceId));
        deviceMetrics.updateMetrics(metrics);
    }
    
    /**
     * Update node metrics
     * 
     * @param nodeId Node identifier
     * @param metrics Node performance metrics
     */
    public void updateNodeMetrics(String nodeId, PerformanceMetrics metrics) {
        NodeMetrics nodeMetrics = this.nodeMetrics.computeIfAbsent(nodeId, 
            k -> new NodeMetrics(nodeId));
        nodeMetrics.updateMetrics(metrics);
    }
    
    /**
     * Update network metrics
     * 
     * @param statistics Network statistics
     */
    public void updateNetworkMetrics(NetworkStatistics statistics) {
        this.networkStatistics = statistics;
    }
    
    /**
     * Update device health
     * 
     * @param healthPercentage Health percentage (0-100)
     */
    public void updateDeviceHealth(double healthPercentage) {
        totalDevicesActive.incrementAndGet();
        logger.debug("Device health updated: {}%", healthPercentage);
    }
    
    /**
     * Update edge health
     * 
     * @param healthPercentage Health percentage (0-100)
     */
    public void updateEdgeHealth(double healthPercentage) {
        totalNodesActive.incrementAndGet();
        logger.debug("Edge health updated: {}%", healthPercentage);
    }
    
    /**
     * Update transmission statistics
     * 
     * @param totalData Total data transmitted
     * @param successful Successful transmissions
     * @param failed Failed transmissions
     * @param successRate Success rate
     */
    public void updateTransmissionStats(long totalData, long successful, long failed, double successRate) {
        totalDataProcessed.addAndGet(totalData);
        logger.debug("Transmission stats: {} bytes, {} successful, {} failed, {}% success",
            totalData, successful, failed, successRate * 100);
    }
    
    /**
     * Get average latency
     * 
     * @return Average latency in milliseconds
     */
    public double getAverageLatency() {
        return averageLatency.get();
    }
    
    /**
     * Get latency reduction
     * 
     * @return Latency reduction percentage
     */
    public double getLatencyReduction() {
        return latencyReduction.get();
    }
    
    /**
     * Get data reduction at edge
     * 
     * @return Data reduction percentage
     */
    public double getDataReductionAtEdge() {
        return dataReductionAtEdge.get();
    }
    
    /**
     * Get energy efficiency
     * 
     * @return Energy efficiency percentage
     */
    public double getEnergyEfficiency() {
        return energyEfficiency.get();
    }
    
    /**
     * Get bandwidth optimization
     * 
     * @return Bandwidth optimization percentage
     */
    public double getBandwidthOptimization() {
        return bandwidthOptimization.get();
    }
    
    /**
     * Get all metrics
     * 
     * @return System metrics
     */
    public SystemMetrics getMetrics() {
        return new SystemMetrics(
            totalDataProcessed.get(),
            totalDevicesActive.get(),
            totalNodesActive.get(),
            averageLatency.get(),
            latencyReduction.get(),
            dataReductionAtEdge.get(),
            energyEfficiency.get(),
            bandwidthOptimization.get(),
            networkStatistics,
            deviceMetrics,
            nodeMetrics
        );
    }
    
    /**
     * Calculate aggregate metrics
     */
    private void calculateAggregateMetrics() {
        // Calculate average latency from all devices and nodes
        double totalLatency = 0.0;
        int latencyCount = 0;
        
        for (DeviceMetrics metrics : deviceMetrics.values()) {
            totalLatency += metrics.getAverageLatency();
            latencyCount++;
        }
        
        for (NodeMetrics metrics : nodeMetrics.values()) {
            totalLatency += metrics.getAverageLatency();
            latencyCount++;
        }
        
        if (latencyCount > 0) {
            averageLatency.set(totalLatency / latencyCount);
        }
        
        // Calculate other aggregate metrics
        calculateLatencyReduction();
        calculateDataReduction();
        calculateEnergyEfficiency();
        calculateBandwidthOptimization();
    }
    
    /**
     * Calculate latency reduction compared to cloud-only processing
     */
    private void calculateLatencyReduction() {
        // Simulate cloud-only latency (higher) vs edge processing (lower)
        double cloudOnlyLatency = 200.0; // Simulated cloud latency
        double currentLatency = averageLatency.get();
        
        if (cloudOnlyLatency > 0) {
            double reduction = ((cloudOnlyLatency - currentLatency) / cloudOnlyLatency) * 100.0;
            latencyReduction.set(Math.max(0.0, Math.min(100.0, reduction)));
        }
    }
    
    /**
     * Calculate data reduction at edge
     */
    private void calculateDataReduction() {
        // Simulate data reduction through edge processing
        double originalDataSize = 1000.0; // Simulated original data size
        double processedDataSize = 300.0; // Simulated processed data size
        
        if (originalDataSize > 0) {
            double reduction = ((originalDataSize - processedDataSize) / originalDataSize) * 100.0;
            dataReductionAtEdge.set(Math.max(0.0, Math.min(100.0, reduction)));
        }
    }
    
    /**
     * Calculate energy efficiency
     */
    private void calculateEnergyEfficiency() {
        // Simulate energy efficiency improvement
        double baselineEnergy = 100.0; // Simulated baseline energy consumption
        double currentEnergy = 65.0; // Simulated current energy consumption
        
        if (baselineEnergy > 0) {
            double efficiency = ((baselineEnergy - currentEnergy) / baselineEnergy) * 100.0;
            energyEfficiency.set(Math.max(0.0, Math.min(100.0, efficiency)));
        }
    }
    
    /**
     * Calculate bandwidth optimization
     */
    private void calculateBandwidthOptimization() {
        // Simulate bandwidth optimization
        double originalBandwidth = 100.0; // Simulated original bandwidth usage
        double optimizedBandwidth = 45.0; // Simulated optimized bandwidth usage
        
        if (originalBandwidth > 0) {
            double optimization = ((originalBandwidth - optimizedBandwidth) / originalBandwidth) * 100.0;
            bandwidthOptimization.set(Math.max(0.0, Math.min(100.0, optimization)));
        }
    }
    
    /**
     * Update performance indicators
     */
    private void updatePerformanceIndicators() {
        // Update performance metrics based on current system state
        logger.debug("Performance indicators updated - Latency: {:.2f}ms, Reduction: {:.2f}%, Data Reduction: {:.2f}%",
            averageLatency.get(), latencyReduction.get(), dataReductionAtEdge.get());
    }
    
    /**
     * Get device count
     * 
     * @return Number of active devices
     */
    public long getActiveDeviceCount() {
        return totalDevicesActive.get();
    }
    
    /**
     * Get node count
     * 
     * @return Number of active nodes
     */
    public long getActiveNodeCount() {
        return totalNodesActive.get();
    }
    
    /**
     * Get total data processed
     * 
     * @return Total data processed in bytes
     */
    public long getTotalDataProcessed() {
        return totalDataProcessed.get();
    }
    
    /**
     * Update edge node metrics
     * 
     * @param nodeId Node identifier
     * @param metrics Node metrics
     */
    public void updateEdgeNodeMetrics(String nodeId, Map<String, Object> metrics) {
        NodeMetrics nodeMetrics = this.nodeMetrics.computeIfAbsent(nodeId, 
            k -> new NodeMetrics(nodeId));
        nodeMetrics.updateMetrics(new PerformanceMetrics(nodeId, "EDGE_NODE"));
        logger.debug("Updated edge node metrics for: {}", nodeId);
    }
    
    /**
     * Update cloud service metrics
     * 
     * @param serviceId Service identifier
     * @param metrics Service metrics
     */
    public void updateCloudServiceMetrics(String serviceId, Map<String, Object> metrics) {
        // Create a new performance metrics object for cloud service
        PerformanceMetrics performanceMetrics = new PerformanceMetrics(serviceId, "CLOUD_SERVICE");
        for (Map.Entry<String, Object> entry : metrics.entrySet()) {
            performanceMetrics.addMetric(entry.getKey(), entry.getValue());
        }
        logger.debug("Updated cloud service metrics for: {}", serviceId);
    }
    
    /**
     * Update cloud health
     * 
     * @param healthPercentage Health percentage
     */
    public void updateCloudHealth(double healthPercentage) {
        logger.debug("Cloud health updated: {}%", healthPercentage);
    }
    
    /**
     * Update cloud processing stats
     * 
     * @param tasksProcessed Number of tasks processed
     * @param processingTime Processing time in milliseconds
     */
    public void updateCloudProcessingStats(int tasksProcessed, double processingTime) {
        logger.debug("Cloud processing stats: {} tasks, {}ms", tasksProcessed, processingTime);
    }
    
    /**
     * Update task reception stats
     * 
     * @param tasksReceived Number of tasks received
     * @param tasksProcessed Number of tasks processed
     * @param successRate Success rate percentage
     */
    public void updateTaskReceptionStats(int tasksReceived, int tasksProcessed, double successRate) {
        logger.debug("Task reception stats: {} received, {} processed, {}% success", 
            tasksReceived, tasksProcessed, successRate);
    }
    
    /**
     * Update processing stats
     * 
     * @param dataProcessed Data processed in bytes
     * @param processingTime Processing time in milliseconds
     */
    public void updateProcessingStats(int dataProcessed, double processingTime) {
        logger.debug("Processing stats: {} bytes, {}ms", dataProcessed, processingTime);
    }
    
    /**
     * Update offloading stats
     * 
     * @param tasksOffloaded Number of tasks offloaded
     * @param totalTasks Total number of tasks
     * @param offloadingRate Offloading rate percentage
     */
    public void updateOffloadingStats(int tasksOffloaded, int totalTasks, double offloadingRate) {
        logger.debug("Offloading stats: {} offloaded, {} total, {}% rate", 
            tasksOffloaded, totalTasks, offloadingRate);
    }
} 