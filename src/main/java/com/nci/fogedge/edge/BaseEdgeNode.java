package com.nci.fogedge.edge;

import com.nci.fogedge.network.NetworkManager;
import com.nci.fogedge.utils.MetricsCollector;
import com.nci.fogedge.utils.DiagnosticResult;
import com.nci.fogedge.utils.PerformanceMetrics;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Base abstract class for Edge Nodes in the Fog and Edge Computing System
 * 
 * This class provides common functionality for all edge nodes including
 * lifecycle management, data processing, performance tracking, and
 * intelligent task offloading decisions.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public abstract class BaseEdgeNode implements EdgeNode {
    
    protected static final Logger logger = LoggerFactory.getLogger(BaseEdgeNode.class);
    
    // Node properties
    protected final String nodeId;
    protected final String nodeType;
    protected volatile String status;
    protected volatile boolean isRunning;
    
    // Dependencies
    protected final NetworkManager networkManager;
    protected final MetricsCollector metricsCollector;
    
    // Performance tracking
    protected final AtomicLong totalDataProcessed;
    protected final AtomicInteger tasksOffloaded;
    protected final AtomicInteger processingTimeTotal;
    protected final AtomicInteger processingCount;
    protected final AtomicInteger errorCount;
    protected volatile long lastDataProcessingTime;
    protected volatile long lastTaskOffloadingTime;
    protected volatile long startTime;
    
    // Resource utilization
    protected volatile double cpuUtilization;
    protected volatile double memoryUtilization;
    protected volatile double bandwidthUtilization;
    
    // Configuration
    protected final Map<String, Object> configuration;
    
    // Thread management
    protected ScheduledExecutorService nodeExecutor;
    protected ScheduledFuture<?> dataProcessingTask;
    protected ScheduledFuture<?> healthCheckTask;
    protected ScheduledFuture<?> resourceMonitorTask;
    
    /**
     * Constructor for base edge node
     * 
     * @param nodeId Unique node identifier
     * @param nodeType Type of node (e.g., "DATA_PROCESSING")
     * @param networkManager Network manager for communication
     * @param metricsCollector Metrics collector for performance tracking
     */
    protected BaseEdgeNode(String nodeId, String nodeType, 
                          NetworkManager networkManager, 
                          MetricsCollector metricsCollector) {
        this.nodeId = nodeId;
        this.nodeType = nodeType;
        this.networkManager = networkManager;
        this.metricsCollector = metricsCollector;
        
        this.status = "INACTIVE";
        this.isRunning = false;
        
        this.totalDataProcessed = new AtomicLong(0);
        this.tasksOffloaded = new AtomicInteger(0);
        this.processingTimeTotal = new AtomicInteger(0);
        this.processingCount = new AtomicInteger(0);
        this.errorCount = new AtomicInteger(0);
        this.lastDataProcessingTime = 0;
        this.lastTaskOffloadingTime = 0;
        this.startTime = System.currentTimeMillis();
        
        this.cpuUtilization = 20.0; // Start with low utilization
        this.memoryUtilization = 30.0; // Start with moderate memory usage
        this.bandwidthUtilization = 15.0; // Start with low bandwidth usage
        
        this.configuration = new ConcurrentHashMap<>();
        initializeDefaultConfiguration();
        
        logger.debug("Base edge node initialized: {}", nodeId);
    }
    
    /**
     * Initialize default configuration for the node
     */
    protected void initializeDefaultConfiguration() {
        configuration.put("maxCpuUtilization", 80.0); // Maximum CPU utilization before offloading
        configuration.put("maxMemoryUtilization", 85.0); // Maximum memory utilization
        configuration.put("maxBandwidthUtilization", 75.0); // Maximum bandwidth utilization
        configuration.put("processingInterval", 5); // seconds
        configuration.put("offloadingThreshold", 0.7); // 70% threshold for offloading
        configuration.put("maxProcessingTime", 1000); // milliseconds
    }
    
    @Override
    public String getNodeId() {
        return nodeId;
    }
    
    @Override
    public String getNodeType() {
        return nodeType;
    }
    
    @Override
    public String getStatus() {
        return status;
    }
    
    @Override
    public boolean isHealthy() {
        return isRunning && cpuUtilization < 90.0 && memoryUtilization < 95.0;
    }
    
    @Override
    public void start() {
        if (isRunning) {
            logger.warn("Edge node {} is already running", nodeId);
            return;
        }
        
        logger.info("Starting edge node: {}", nodeId);
        
        try {
            // Initialize node-specific components
            initializeNode();
            
            // Create executor for node tasks
            nodeExecutor = Executors.newScheduledThreadPool(3);
            
            // Start data processing task
            int interval = (Integer) configuration.get("processingInterval");
            dataProcessingTask = nodeExecutor.scheduleAtFixedRate(() -> {
                try {
                    processIncomingData();
                } catch (Exception e) {
                    logger.error("Error in data processing for node: {}", nodeId, e);
                }
            }, 0, interval, TimeUnit.SECONDS);
            
            // Start health check task
            healthCheckTask = nodeExecutor.scheduleAtFixedRate(() -> {
                try {
                    performHealthCheck();
                } catch (Exception e) {
                    logger.error("Error in health check for node: {}", nodeId, e);
                }
            }, 10, 60, TimeUnit.SECONDS);
            
            // Start resource monitoring task
            resourceMonitorTask = nodeExecutor.scheduleAtFixedRate(() -> {
                try {
                    monitorResources();
                } catch (Exception e) {
                    logger.error("Error in resource monitoring for node: {}", nodeId, e);
                }
            }, 5, 30, TimeUnit.SECONDS);
            
            isRunning = true;
            status = "ACTIVE";
            
            logger.info("Edge node {} started successfully", nodeId);
            
        } catch (Exception e) {
            logger.error("Failed to start edge node: {}", nodeId, e);
            status = "ERROR";
            throw new RuntimeException("Edge node startup failed", e);
        }
    }
    
    @Override
    public void stop() {
        if (!isRunning) {
            logger.warn("Edge node {} is not running", nodeId);
            return;
        }
        
        logger.info("Stopping edge node: {}", nodeId);
        
        try {
            // Stop scheduled tasks
            if (dataProcessingTask != null) {
                dataProcessingTask.cancel(true);
            }
            if (healthCheckTask != null) {
                healthCheckTask.cancel(true);
            }
            if (resourceMonitorTask != null) {
                resourceMonitorTask.cancel(true);
            }
            
            // Shutdown executor
            if (nodeExecutor != null) {
                nodeExecutor.shutdown();
                if (!nodeExecutor.awaitTermination(10, TimeUnit.SECONDS)) {
                    nodeExecutor.shutdownNow();
                }
            }
            
            // Perform node-specific cleanup
            cleanupNode();
            
            isRunning = false;
            status = "INACTIVE";
            
            logger.info("Edge node {} stopped successfully", nodeId);
            
        } catch (Exception e) {
            logger.error("Error stopping edge node: {}", nodeId, e);
            status = "ERROR";
        }
    }
    
    public double getCpuUtilization() {
        return cpuUtilization;
    }
    
    public double getMemoryUtilization() {
        return memoryUtilization;
    }
    
    public double getBandwidthUtilization() {
        return bandwidthUtilization;
    }
    
    public long getTotalDataProcessed() {
        return totalDataProcessed.get();
    }
    
    public int getTasksOffloaded() {
        return tasksOffloaded.get();
    }
    
    public double getAverageProcessingTime() {
        int count = processingCount.get();
        return count > 0 ? (double) processingTimeTotal.get() / count : 0;
    }
    
    public double getOffloadingRate() {
        int total = processingCount.get();
        return total > 0 ? (double) tasksOffloaded.get() / total * 100 : 0;
    }
    
    public Map<String, Object> getConfiguration() {
        return new HashMap<>(configuration);
    }
    
    @Override
    public void updateConfiguration(Map<String, Object> config) {
        configuration.putAll(config);
        logger.info("Configuration updated for edge node: {}", nodeId);
    }
    
    public Map<String, Object> getPerformanceMetrics() {
        Map<String, Object> metrics = new HashMap<>();
        metrics.put("nodeId", nodeId);
        metrics.put("nodeType", nodeType);
        metrics.put("status", status);
        metrics.put("cpuUtilization", cpuUtilization);
        metrics.put("memoryUtilization", memoryUtilization);
        metrics.put("bandwidthUtilization", bandwidthUtilization);
        metrics.put("totalDataProcessed", totalDataProcessed.get());
        metrics.put("tasksOffloaded", tasksOffloaded.get());
        metrics.put("averageProcessingTime", getAverageProcessingTime());
        metrics.put("offloadingRate", getOffloadingRate());
        
        return metrics;
    }
    
    public void resetStatistics() {
        totalDataProcessed.set(0);
        tasksOffloaded.set(0);
        processingTimeTotal.set(0);
        processingCount.set(0);
        
        logger.info("Statistics reset for edge node: {}", nodeId);
    }
    
    public DiagnosticResult performDiagnostic() {
        logger.debug("Performing diagnostic for edge node: {}", nodeId);
        
        Map<String, Object> details = new HashMap<>();
        boolean passed = true;
        String message = "Diagnostic passed";
        
        // Check CPU utilization
        if (cpuUtilization > 90.0) {
            passed = false;
            message = "High CPU utilization";
        }
        details.put("cpuUtilization", cpuUtilization);
        
        // Check memory utilization
        if (memoryUtilization > 95.0) {
            passed = false;
            message = "High memory utilization";
        }
        details.put("memoryUtilization", memoryUtilization);
        
        // Check bandwidth utilization
        if (bandwidthUtilization > 90.0) {
            passed = false;
            message = "High bandwidth utilization";
        }
        details.put("bandwidthUtilization", bandwidthUtilization);
        
        // Check node status
        details.put("status", status);
        details.put("isRunning", isRunning);
        
        return new DiagnosticResult(passed, message, details);
    }
    
    public boolean shouldOffloadTask(Object task) {
        // Implement intelligent offloading decision based on:
        // 1. Current resource utilization
        // 2. Task complexity
        // 3. Network conditions
        // 4. Processing queue length
        
        double offloadingThreshold = (Double) configuration.get("offloadingThreshold");
        
        // Check resource utilization
        boolean highCpu = cpuUtilization > (Double) configuration.get("maxCpuUtilization");
        boolean highMemory = memoryUtilization > (Double) configuration.get("maxMemoryUtilization");
        boolean highBandwidth = bandwidthUtilization > (Double) configuration.get("maxBandwidthUtilization");
        
        // Check if task is complex (simplified heuristic)
        boolean complexTask = isComplexTask(task);
        
        // Decision logic
        if (highCpu || highMemory || (complexTask && highBandwidth)) {
            return Math.random() < offloadingThreshold;
        }
        
        return false;
    }
    
    public boolean offloadTask(Object task) {
        try {
            logger.debug("Offloading task from edge node: {}", nodeId);
            
            // Simulate task offloading to cloud
            boolean offloadingSuccess = networkManager.offloadTaskToCloud(nodeId, task);
            
            if (offloadingSuccess) {
                tasksOffloaded.incrementAndGet();
                logger.debug("Task offloaded successfully from edge node: {}", nodeId);
            } else {
                logger.warn("Task offloading failed from edge node: {}", nodeId);
            }
            
            return offloadingSuccess;
            
        } catch (Exception e) {
            logger.error("Error offloading task from edge node: {}", nodeId, e);
            return false;
        }
    }
    
    /**
     * Process incoming data from IoT devices
     */
    protected void processIncomingData() {
        try {
            Object incomingData = networkManager.receiveDataFromIoT();
            if (incomingData != null) {
                long startTime = System.currentTimeMillis();
                if (shouldOffloadTask(incomingData)) {
                    offloadTask(incomingData);
                } else {
                    // Process data locally
                    String dataString = incomingData.toString();
                    Object result = processData(dataString);
                    long processingTime = System.currentTimeMillis() - startTime;
                    processingTimeTotal.addAndGet((int) processingTime);
                    processingCount.incrementAndGet();
                    totalDataProcessed.addAndGet(dataString.getBytes().length);
                    logger.debug("Data processed locally by edge node: {} in {}ms", nodeId, processingTime);
                }
            }
        } catch (Exception e) {
            logger.error("Error processing incoming data for edge node: {}", nodeId, e);
        }
    }
    
    /**
     * Perform periodic health check
     */
    protected void performHealthCheck() {
        DiagnosticResult result = performDiagnostic();
        
        if (!result.isPassed()) {
            logger.warn("Health check failed for edge node {}: {}", nodeId, result.getMessage());
            status = "WARNING";
        } else {
            status = "ACTIVE";
        }
        
        // Update metrics
        metricsCollector.updateEdgeNodeMetrics(nodeId, getPerformanceMetrics());
    }
    
    /**
     * Monitor resource utilization
     */
    protected void monitorResources() {
        // Simulate resource monitoring with realistic variations
        double cpuVariation = (Math.random() - 0.5) * 10.0;
        double memoryVariation = (Math.random() - 0.5) * 5.0;
        double bandwidthVariation = (Math.random() - 0.5) * 8.0;
        
        cpuUtilization = Math.max(0.0, Math.min(100.0, cpuUtilization + cpuVariation));
        memoryUtilization = Math.max(0.0, Math.min(100.0, memoryUtilization + memoryVariation));
        bandwidthUtilization = Math.max(0.0, Math.min(100.0, bandwidthUtilization + bandwidthVariation));
        
        logger.debug("Resource utilization for edge node {}: CPU={:.1f}%, Memory={:.1f}%, Bandwidth={:.1f}%", 
                    nodeId, cpuUtilization, memoryUtilization, bandwidthUtilization);
    }
    
    /**
     * Check if a task is complex (simplified heuristic)
     * 
     * @param task Task to evaluate
     * @return true if task is complex, false otherwise
     */
    protected boolean isComplexTask(Object task) {
        // Simplified complexity check based on data size and type
        if (task instanceof Map) {
            Map<?, ?> taskMap = (Map<?, ?>) task;
            return taskMap.size() > 10; // More than 10 fields indicates complexity
        }
        return task.toString().length() > 1000; // Large data indicates complexity
    }
    
    /**
     * Initialize node-specific components
     */
    protected abstract void initializeNode();
    
    /**
     * Cleanup node-specific resources
     */
    protected abstract void cleanupNode();

    @Override
    public double getProcessingCapacity() {
        return 100.0; // MB/s - typical edge node capacity
    }

    @Override
    public double getStorageCapacity() {
        return 1000.0; // MB - typical edge node storage
    }

    public double getCpuUsage() {
        return cpuUtilization;
    }

    public double getMemoryUsage() {
        return memoryUtilization;
    }

    public double getStorageUsage() {
        return 0.0; // Edge nodes typically don't have persistent storage
    }

    public double getBandwidthUsage() {
        return bandwidthUtilization;
    }

    public double getEnergyConsumption() {
        // Estimate energy consumption based on CPU and memory usage
        return cpuUtilization * 0.5 + memoryUtilization * 0.3;
    }

    public double getTaskProcessingRate() {
        if (processingCount.get() == 0) return 0.0;
        return processingCount.get() / (System.currentTimeMillis() - startTime) * 1000.0;
    }

    public double getTaskOffloadingRate() {
        if (tasksOffloaded.get() == 0) return 0.0;
        return tasksOffloaded.get() / (System.currentTimeMillis() - startTime) * 1000.0;
    }

    public int getErrorCount() {
        return errorCount.get();
    }

    public void resetErrorCount() {
        errorCount.set(0);
    }

    public long getLastDataProcessingTime() {
        return lastDataProcessingTime;
    }

    public long getLastTaskOffloadingTime() {
        return lastTaskOffloadingTime;
    }

    @Override
    public PerformanceMetrics getMetrics() {
        PerformanceMetrics metrics = new PerformanceMetrics(nodeId, nodeType);
        metrics.addMetric("cpu_usage", cpuUtilization);
        metrics.addMetric("memory_usage", memoryUtilization);
        metrics.addMetric("bandwidth_usage", bandwidthUtilization);
        metrics.addMetric("task_processing_rate", getTaskProcessingRate());
        metrics.addMetric("task_offloading_rate", getTaskOffloadingRate());
        metrics.addMetric("error_count", errorCount.get());
        metrics.addMetric("healthy", isHealthy());
        metrics.addMetric("running", isRunning());
        return metrics;
    }
} 