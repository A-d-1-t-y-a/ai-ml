package com.nci.fogedge.cloud;

import com.nci.fogedge.network.NetworkManager;
import com.nci.fogedge.utils.MetricsCollector;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Base abstract class for Cloud Services in the Fog and Edge Computing System
 * 
 * This class provides common functionality for all cloud services including
 * lifecycle management, task processing, performance tracking, and resource monitoring.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public abstract class BaseCloudService implements CloudService {
    
    protected static final Logger logger = LoggerFactory.getLogger(BaseCloudService.class);
    
    // Service properties
    protected final String serviceId;
    protected final String serviceType;
    protected volatile String status;
    protected volatile boolean isRunning;
    
    // Dependencies
    protected final NetworkManager networkManager;
    protected final MetricsCollector metricsCollector;
    
    // Performance tracking
    protected final AtomicInteger totalTasksProcessed;
    protected final AtomicInteger processingTimeTotal;
    protected final AtomicInteger queueLength;
    
    // Resource utilization
    protected volatile double cpuUtilization;
    protected volatile double memoryUtilization;
    protected volatile double bandwidthUtilization;
    
    // Configuration
    protected final Map<String, Object> configuration;
    
    // Thread management
    protected ScheduledExecutorService serviceExecutor;
    protected ScheduledFuture<?> taskProcessingTask;
    protected ScheduledFuture<?> healthCheckTask;
    protected ScheduledFuture<?> resourceMonitorTask;
    
    /**
     * Constructor for base cloud service
     * 
     * @param serviceId Unique service identifier
     * @param serviceType Type of service (e.g., "DATA_ANALYTICS")
     * @param networkManager Network manager for communication
     * @param metricsCollector Metrics collector for performance tracking
     */
    protected BaseCloudService(String serviceId, String serviceType, 
                              NetworkManager networkManager, 
                              MetricsCollector metricsCollector) {
        this.serviceId = serviceId;
        this.serviceType = serviceType;
        this.networkManager = networkManager;
        this.metricsCollector = metricsCollector;
        
        this.status = "INACTIVE";
        this.isRunning = false;
        
        this.totalTasksProcessed = new AtomicInteger(0);
        this.processingTimeTotal = new AtomicInteger(0);
        this.queueLength = new AtomicInteger(0);
        
        this.cpuUtilization = 15.0; // Start with low utilization
        this.memoryUtilization = 25.0; // Start with moderate memory usage
        this.bandwidthUtilization = 10.0; // Start with low bandwidth usage
        
        this.configuration = new ConcurrentHashMap<>();
        initializeDefaultConfiguration();
        
        logger.debug("Base cloud service initialized: {}", serviceId);
    }
    
    /**
     * Initialize default configuration for the service
     */
    protected void initializeDefaultConfiguration() {
        configuration.put("maxCpuUtilization", 85.0); // Maximum CPU utilization
        configuration.put("maxMemoryUtilization", 90.0); // Maximum memory utilization
        configuration.put("maxBandwidthUtilization", 80.0); // Maximum bandwidth utilization
        configuration.put("processingInterval", 10); // seconds
        configuration.put("maxQueueLength", 1000);
        configuration.put("timeout", 30000); // milliseconds
    }
    
    @Override
    public String getServiceId() {
        return serviceId;
    }
    
    @Override
    public String getServiceType() {
        return serviceType;
    }
    
    @Override
    public String getStatus() {
        return status;
    }
    
    @Override
    public boolean isHealthy() {
        return isRunning && cpuUtilization < 95.0 && memoryUtilization < 98.0 && queueLength.get() < 500;
    }
    
    @Override
    public void start() {
        if (isRunning) {
            logger.warn("Cloud service {} is already running", serviceId);
            return;
        }
        
        logger.info("Starting cloud service: {}", serviceId);
        
        try {
            // Initialize service-specific components
            initializeService();
            
            // Create executor for service tasks
            serviceExecutor = Executors.newScheduledThreadPool(5);
            
            // Start task processing task
            int interval = (Integer) configuration.get("processingInterval");
            taskProcessingTask = serviceExecutor.scheduleAtFixedRate(() -> {
                try {
                    processIncomingTasks();
                } catch (Exception e) {
                    logger.error("Error in task processing for service: {}", serviceId, e);
                }
            }, 0, interval, TimeUnit.SECONDS);
            
            // Start health check task
            healthCheckTask = serviceExecutor.scheduleAtFixedRate(() -> {
                try {
                    performHealthCheck();
                } catch (Exception e) {
                    logger.error("Error in health check for service: {}", serviceId, e);
                }
            }, 10, 60, TimeUnit.SECONDS);
            
            // Start resource monitoring task
            resourceMonitorTask = serviceExecutor.scheduleAtFixedRate(() -> {
                try {
                    monitorResources();
                } catch (Exception e) {
                    logger.error("Error in resource monitoring for service: {}", serviceId, e);
                }
            }, 5, 30, TimeUnit.SECONDS);
            
            isRunning = true;
            status = "ACTIVE";
            
            logger.info("Cloud service {} started successfully", serviceId);
            
        } catch (Exception e) {
            logger.error("Failed to start cloud service: {}", serviceId, e);
            status = "ERROR";
            throw new RuntimeException("Cloud service startup failed", e);
        }
    }
    
    @Override
    public void stop() {
        if (!isRunning) {
            logger.warn("Cloud service {} is not running", serviceId);
            return;
        }
        
        logger.info("Stopping cloud service: {}", serviceId);
        
        try {
            // Stop scheduled tasks
            if (taskProcessingTask != null) {
                taskProcessingTask.cancel(true);
            }
            if (healthCheckTask != null) {
                healthCheckTask.cancel(true);
            }
            if (resourceMonitorTask != null) {
                resourceMonitorTask.cancel(true);
            }
            
            // Shutdown executor
            if (serviceExecutor != null) {
                serviceExecutor.shutdown();
                if (!serviceExecutor.awaitTermination(10, TimeUnit.SECONDS)) {
                    serviceExecutor.shutdownNow();
                }
            }
            
            // Perform service-specific cleanup
            cleanupService();
            
            isRunning = false;
            status = "INACTIVE";
            
            logger.info("Cloud service {} stopped successfully", serviceId);
            
        } catch (Exception e) {
            logger.error("Error stopping cloud service: {}", serviceId, e);
            status = "ERROR";
        }
    }
    
    @Override
    public double getCpuUtilization() {
        return cpuUtilization;
    }
    
    @Override
    public double getMemoryUtilization() {
        return memoryUtilization;
    }
    
    @Override
    public double getBandwidthUtilization() {
        return bandwidthUtilization;
    }
    
    @Override
    public int getTotalTasksProcessed() {
        return totalTasksProcessed.get();
    }
    
    @Override
    public double getAverageProcessingTime() {
        int count = totalTasksProcessed.get();
        return count > 0 ? (double) processingTimeTotal.get() / count : 0;
    }
    
    @Override
    public double getServiceEfficiency() {
        // Calculate efficiency based on resource utilization and processing performance
        double resourceEfficiency = (100.0 - cpuUtilization) * 0.4 + 
                                   (100.0 - memoryUtilization) * 0.3 + 
                                   (100.0 - bandwidthUtilization) * 0.3;
        
        double processingEfficiency = Math.min(100.0, 1000.0 / Math.max(1.0, getAverageProcessingTime()));
        
        return (resourceEfficiency + processingEfficiency) / 2.0;
    }
    
    @Override
    public Map<String, Object> getConfiguration() {
        return new HashMap<>(configuration);
    }
    
    @Override
    public void updateConfiguration(Map<String, Object> config) {
        configuration.putAll(config);
        logger.info("Configuration updated for cloud service: {}", serviceId);
    }
    
    @Override
    public Map<String, Object> getPerformanceMetrics() {
        Map<String, Object> metrics = new HashMap<>();
        metrics.put("serviceId", serviceId);
        metrics.put("serviceType", serviceType);
        metrics.put("status", status);
        metrics.put("cpuUtilization", cpuUtilization);
        metrics.put("memoryUtilization", memoryUtilization);
        metrics.put("bandwidthUtilization", bandwidthUtilization);
        metrics.put("totalTasksProcessed", totalTasksProcessed.get());
        metrics.put("averageProcessingTime", getAverageProcessingTime());
        metrics.put("serviceEfficiency", getServiceEfficiency());
        metrics.put("queueLength", queueLength.get());
        
        return metrics;
    }
    
    @Override
    public void resetStatistics() {
        totalTasksProcessed.set(0);
        processingTimeTotal.set(0);
        queueLength.set(0);
        
        logger.info("Statistics reset for cloud service: {}", serviceId);
    }
    
    @Override
    public DiagnosticResult performDiagnostic() {
        logger.debug("Performing diagnostic for cloud service: {}", serviceId);
        
        Map<String, Object> details = new HashMap<>();
        boolean passed = true;
        String message = "Diagnostic passed";
        
        // Check CPU utilization
        if (cpuUtilization > 95.0) {
            passed = false;
            message = "High CPU utilization";
        }
        details.put("cpuUtilization", cpuUtilization);
        
        // Check memory utilization
        if (memoryUtilization > 98.0) {
            passed = false;
            message = "High memory utilization";
        }
        details.put("memoryUtilization", memoryUtilization);
        
        // Check bandwidth utilization
        if (bandwidthUtilization > 95.0) {
            passed = false;
            message = "High bandwidth utilization";
        }
        details.put("bandwidthUtilization", bandwidthUtilization);
        
        // Check queue length
        if (queueLength.get() > 500) {
            passed = false;
            message = "Long processing queue";
        }
        details.put("queueLength", queueLength.get());
        
        // Check service status
        details.put("status", status);
        details.put("isRunning", isRunning);
        
        return new DiagnosticResult(passed, message, details);
    }
    
    @Override
    public boolean canAcceptTasks() {
        return isRunning && 
               cpuUtilization < (Double) configuration.get("maxCpuUtilization") &&
               memoryUtilization < (Double) configuration.get("maxMemoryUtilization") &&
               queueLength.get() < (Integer) configuration.get("maxQueueLength");
    }
    
    @Override
    public int getQueueLength() {
        return queueLength.get();
    }
    
    /**
     * Process incoming tasks from edge nodes
     */
    protected void processIncomingTasks() {
        try {
            // Simulate receiving tasks from edge nodes
            Object incomingTask = networkManager.receiveTaskFromEdge();
            
            if (incomingTask != null && canAcceptTasks()) {
                queueLength.incrementAndGet();
                
                long startTime = System.currentTimeMillis();
                
                // Process the task
                Object result = processTask(incomingTask);
                
                // Update statistics
                long processingTime = System.currentTimeMillis() - startTime;
                processingTimeTotal.addAndGet((int) processingTime);
                totalTasksProcessed.incrementAndGet();
                queueLength.decrementAndGet();
                
                logger.debug("Task processed by cloud service: {} in {}ms", serviceId, processingTime);
            }
            
        } catch (Exception e) {
            logger.error("Error processing incoming tasks for cloud service: {}", serviceId, e);
        }
    }
    
    /**
     * Perform periodic health check
     */
    protected void performHealthCheck() {
        DiagnosticResult result = performDiagnostic();
        
        if (!result.isPassed()) {
            logger.warn("Health check failed for cloud service {}: {}", serviceId, result.getMessage());
            status = "WARNING";
        } else {
            status = "ACTIVE";
        }
        
        // Update metrics
        metricsCollector.updateCloudServiceMetrics(serviceId, getPerformanceMetrics());
    }
    
    /**
     * Monitor resource utilization
     */
    protected void monitorResources() {
        // Simulate resource monitoring with realistic variations
        double cpuVariation = (Math.random() - 0.5) * 15.0;
        double memoryVariation = (Math.random() - 0.5) * 8.0;
        double bandwidthVariation = (Math.random() - 0.5) * 12.0;
        
        cpuUtilization = Math.max(0.0, Math.min(100.0, cpuUtilization + cpuVariation));
        memoryUtilization = Math.max(0.0, Math.min(100.0, memoryUtilization + memoryVariation));
        bandwidthUtilization = Math.max(0.0, Math.min(100.0, bandwidthUtilization + bandwidthVariation));
        
        logger.debug("Resource utilization for cloud service {}: CPU={:.1f}%, Memory={:.1f}%, Bandwidth={:.1f}%", 
                    serviceId, cpuUtilization, memoryUtilization, bandwidthUtilization);
    }
    
    /**
     * Initialize service-specific components
     */
    protected abstract void initializeService();
    
    /**
     * Cleanup service-specific resources
     */
    protected abstract void cleanupService();
} 