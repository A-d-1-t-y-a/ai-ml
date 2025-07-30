package com.nci.fogedge.cloud;

import com.nci.fogedge.network.NetworkManager;
import com.nci.fogedge.utils.MetricsCollector;
import com.nci.fogedge.utils.ConfigurationManager;
import com.nci.fogedge.cloud.services.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Cloud Manager for the Fog and Edge Computing System
 * 
 * This class manages cloud computing services that receive offloaded tasks
 * from edge nodes and perform heavy computational tasks, analytics, and
 * data storage. Based on the research paper's cloud layer implementation.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class CloudManager {
    
    private static final Logger logger = LoggerFactory.getLogger(CloudManager.class);
    
    // Configuration and dependencies
    private final ConfigurationManager configManager;
    private final NetworkManager networkManager;
    private final MetricsCollector metricsCollector;
    
    // Cloud service management
    private final Map<String, CloudService> cloudServices;
    private final List<CloudService> activeServices;
    private final AtomicInteger serviceCounter;
    
    // Thread management
    private final ScheduledExecutorService cloudExecutor;
    private final List<Future<?>> cloudTasks;
    
    // Performance tracking
    private final AtomicInteger totalTasksProcessed;
    private final AtomicInteger tasksReceivedFromEdge;
    private final AtomicInteger cloudProcessingTime;
    
    /**
     * Constructor for Cloud Manager
     * 
     * @param configManager Configuration manager for system settings
     * @param networkManager Network manager for communication
     * @param metricsCollector Metrics collector for performance tracking
     */
    public CloudManager(ConfigurationManager configManager, 
                       NetworkManager networkManager, 
                       MetricsCollector metricsCollector) {
        this.configManager = configManager;
        this.networkManager = networkManager;
        this.metricsCollector = metricsCollector;
        
        this.cloudServices = new ConcurrentHashMap<>();
        this.activeServices = Collections.synchronizedList(new ArrayList<>());
        this.serviceCounter = new AtomicInteger(0);
        
        this.cloudExecutor = Executors.newScheduledThreadPool(20);
        this.cloudTasks = Collections.synchronizedList(new ArrayList<>());
        
        this.totalTasksProcessed = new AtomicInteger(0);
        this.tasksReceivedFromEdge = new AtomicInteger(0);
        this.cloudProcessingTime = new AtomicInteger(0);
        
        logger.info("Cloud Manager initialized");
    }
    
    /**
     * Start the Cloud Manager and initialize cloud services
     */
    public void start() {
        logger.info("Starting Cloud Manager...");
        
        try {
            // Create and initialize cloud services
            createCloudServices();
            
            // Start all cloud services
            startAllCloudServices();
            
            // Start cloud monitoring
            startCloudMonitoring();
            
            logger.info("Cloud Manager started successfully with {} services", activeServices.size());
            
        } catch (Exception e) {
            logger.error("Failed to start Cloud Manager", e);
            throw new RuntimeException("Cloud Manager startup failed", e);
        }
    }
    
    /**
     * Create various types of cloud services
     */
    private void createCloudServices() {
        logger.info("Creating cloud services...");
        
        // Data analytics services
        for (int i = 0; i < 3; i++) {
            String serviceId = "CLOUD_ANALYTICS_" + String.format("%03d", i);
            DataAnalyticsService service = new DataAnalyticsService(serviceId, networkManager, metricsCollector);
            cloudServices.put(serviceId, service);
            activeServices.add(service);
            logger.debug("Created data analytics cloud service: {}", serviceId);
        }
        
        // Machine learning services
        for (int i = 0; i < 2; i++) {
            String serviceId = "CLOUD_ML_" + String.format("%03d", i);
            MachineLearningService service = new MachineLearningService(serviceId, networkManager, metricsCollector);
            cloudServices.put(serviceId, service);
            activeServices.add(service);
            logger.debug("Created machine learning cloud service: {}", serviceId);
        }
        
        // Storage services
        for (int i = 0; i < 2; i++) {
            String serviceId = "CLOUD_STORAGE_" + String.format("%03d", i);
            StorageService service = new StorageService(serviceId, networkManager, metricsCollector);
            cloudServices.put(serviceId, service);
            activeServices.add(service);
            logger.debug("Created storage cloud service: {}", serviceId);
        }
        
        // Orchestration services
        for (int i = 0; i < 1; i++) {
            String serviceId = "CLOUD_ORCHESTRATION_" + String.format("%03d", i);
            OrchestrationService service = new OrchestrationService(serviceId, networkManager, metricsCollector);
            cloudServices.put(serviceId, service);
            activeServices.add(service);
            logger.debug("Created orchestration cloud service: {}", serviceId);
        }
        
        logger.info("Created {} cloud services successfully", activeServices.size());
    }
    
    /**
     * Start all cloud services
     */
    private void startAllCloudServices() {
        logger.info("Starting all cloud services...");
        
        for (CloudService service : activeServices) {
            try {
                service.start();
                logger.debug("Started cloud service: {}", service.getServiceId());
            } catch (Exception e) {
                logger.error("Failed to start cloud service: {}", service.getServiceId(), e);
            }
        }
        
        logger.info("All cloud services started");
    }
    
    /**
     * Start cloud monitoring and data collection
     */
    private void startCloudMonitoring() {
        logger.info("Starting cloud monitoring...");
        
        // Monitor cloud service health
        Future<?> healthMonitor = cloudExecutor.scheduleAtFixedRate(() -> {
            try {
                monitorCloudHealth();
            } catch (Exception e) {
                logger.error("Error in cloud health monitoring", e);
            }
        }, 10, 60, TimeUnit.SECONDS);
        cloudTasks.add(healthMonitor);
        
        // Monitor processing performance
        Future<?> performanceMonitor = cloudExecutor.scheduleAtFixedRate(() -> {
            try {
                monitorProcessingPerformance();
            } catch (Exception e) {
                logger.error("Error in processing performance monitoring", e);
            }
        }, 15, 45, TimeUnit.SECONDS);
        cloudTasks.add(performanceMonitor);
        
        // Monitor task reception from edge
        Future<?> taskMonitor = cloudExecutor.scheduleAtFixedRate(() -> {
            try {
                monitorTaskReception();
            } catch (Exception e) {
                logger.error("Error in task reception monitoring", e);
            }
        }, 20, 90, TimeUnit.SECONDS);
        cloudTasks.add(taskMonitor);
        
        logger.info("Cloud monitoring started");
    }
    
    /**
     * Monitor the health of all cloud services
     */
    private void monitorCloudHealth() {
        logger.debug("Monitoring cloud service health...");
        
        int healthyServices = 0;
        int totalServices = activeServices.size();
        
        for (CloudService service : activeServices) {
            if (service.isHealthy()) {
                healthyServices++;
            } else {
                logger.warn("Cloud service {} is unhealthy", service.getServiceId());
            }
        }
        
        double healthPercentage = (double) healthyServices / totalServices * 100;
        logger.info("Cloud Service Health Status: {}/{} services healthy ({:.2f}%)", 
                   healthyServices, totalServices, healthPercentage);
        
        // Update metrics
        metricsCollector.updateCloudHealth(healthPercentage);
    }
    
    /**
     * Monitor processing performance
     */
    private void monitorProcessingPerformance() {
        logger.debug("Monitoring cloud processing performance...");
        
        int totalProcessed = totalTasksProcessed.get();
        int totalTime = cloudProcessingTime.get();
        double avgProcessingTime = totalProcessed > 0 ? (double) totalTime / totalProcessed : 0;
        
        logger.info("Cloud Processing Performance Stats:");
        logger.info("  Total Tasks Processed: {}", totalProcessed);
        logger.info("  Average Processing Time: {:.2f} ms", avgProcessingTime);
        logger.info("  Active Cloud Services: {}", activeServices.size());
        
        // Update metrics
        metricsCollector.updateCloudProcessingStats(totalProcessed, avgProcessingTime);
    }
    
    /**
     * Monitor task reception from edge nodes
     */
    private void monitorTaskReception() {
        logger.debug("Monitoring task reception from edge nodes...");
        
        int totalReceived = tasksReceivedFromEdge.get();
        int totalProcessed = totalTasksProcessed.get();
        double processingRate = totalReceived > 0 ? (double) totalProcessed / totalReceived * 100 : 0;
        
        logger.info("Task Reception Stats:");
        logger.info("  Tasks Received from Edge: {}", totalReceived);
        logger.info("  Tasks Processed: {}", totalProcessed);
        logger.info("  Processing Rate: {:.2f}%", processingRate);
        
        // Update metrics
        metricsCollector.updateTaskReceptionStats(totalReceived, totalProcessed, processingRate);
    }
    
    /**
     * Get the count of active cloud services
     * 
     * @return Number of active cloud services
     */
    public int getActiveServiceCount() {
        return activeServices.size();
    }
    
    /**
     * Get a specific cloud service by ID
     * 
     * @param serviceId Cloud service identifier
     * @return Cloud service or null if not found
     */
    public CloudService getCloudService(String serviceId) {
        return cloudServices.get(serviceId);
    }
    
    /**
     * Get all active cloud services
     * 
     * @return List of active cloud services
     */
    public List<CloudService> getAllCloudServices() {
        return new ArrayList<>(activeServices);
    }
    
    /**
     * Get the overall service status
     * 
     * @return Service status string
     */
    public String getServiceStatus() {
        int healthyServices = 0;
        for (CloudService service : activeServices) {
            if (service.isHealthy()) {
                healthyServices++;
            }
        }
        
        double healthPercentage = (double) healthyServices / activeServices.size() * 100;
        
        if (healthPercentage >= 90.0) {
            return "EXCELLENT";
        } else if (healthPercentage >= 75.0) {
            return "GOOD";
        } else if (healthPercentage >= 50.0) {
            return "FAIR";
        } else {
            return "POOR";
        }
    }
    
    /**
     * Record task processing
     * 
     * @param processingTime Processing time in milliseconds
     */
    public void recordTaskProcessing(int processingTime) {
        totalTasksProcessed.incrementAndGet();
        cloudProcessingTime.addAndGet(processingTime);
    }
    
    /**
     * Record task reception from edge
     */
    public void recordTaskReceptionFromEdge() {
        tasksReceivedFromEdge.incrementAndGet();
    }
    
    /**
     * Stop the Cloud Manager
     */
    public void stop() {
        logger.info("Stopping Cloud Manager...");
        
        try {
            // Stop all cloud services
            for (CloudService service : activeServices) {
                try {
                    service.stop();
                    logger.debug("Stopped cloud service: {}", service.getServiceId());
                } catch (Exception e) {
                    logger.error("Error stopping cloud service: {}", service.getServiceId(), e);
                }
            }
            
            // Cancel all monitoring tasks
            for (Future<?> task : cloudTasks) {
                if (!task.isCancelled()) {
                    task.cancel(true);
                }
            }
            
            // Shutdown executor
            cloudExecutor.shutdown();
            if (!cloudExecutor.awaitTermination(30, TimeUnit.SECONDS)) {
                cloudExecutor.shutdownNow();
            }
            
            logger.info("Cloud Manager stopped successfully");
            
        } catch (Exception e) {
            logger.error("Error stopping Cloud Manager", e);
        }
    }
    
    /**
     * Get performance statistics
     * 
     * @return Map containing performance statistics
     */
    public Map<String, Object> getPerformanceStats() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("totalServices", activeServices.size());
        stats.put("totalTasksProcessed", totalTasksProcessed.get());
        stats.put("tasksReceivedFromEdge", tasksReceivedFromEdge.get());
        stats.put("cloudProcessingTime", cloudProcessingTime.get());
        stats.put("processingRate", tasksReceivedFromEdge.get() > 0 ? 
                  (double) totalTasksProcessed.get() / tasksReceivedFromEdge.get() * 100 : 0);
        
        return stats;
    }
} 