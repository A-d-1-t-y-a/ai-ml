package com.nci.fogedge.cloud.services;

import com.nci.fogedge.cloud.CloudService;
import com.nci.fogedge.network.NetworkManager;
import com.nci.fogedge.utils.MetricsCollector;
import com.nci.fogedge.utils.PerformanceMetrics;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Storage Service implementation for the Fog and Edge Computing System
 * 
 * This class provides cloud storage capabilities for the system.
 * Based on the research paper's cloud service implementation.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class StorageService implements CloudService {
    
    private static final Logger logger = LoggerFactory.getLogger(StorageService.class);
    
    private final String serviceId;
    private final NetworkManager networkManager;
    private final MetricsCollector metricsCollector;
    
    private boolean isRunning;
    private boolean isHealthy;
    private final Map<String, String> storage;
    private final Map<String, Object> metrics;
    
    // Performance metrics
    private double cpuUsage;
    private double memoryUsage;
    private double storageUsage;
    private double bandwidthUsage;
    private double energyConsumption;
    private double taskProcessingRate;
    private double dataStorageRate;
    private int errorCount;
    private long lastTaskProcessingTime;
    private long lastDataStorageTime;
    
    /**
     * Constructor for Storage Service
     * 
     * @param serviceId Unique service identifier
     * @param networkManager Network manager for communication
     * @param metricsCollector Metrics collector for performance tracking
     */
    public StorageService(String serviceId, NetworkManager networkManager, MetricsCollector metricsCollector) {
        this.serviceId = serviceId;
        this.networkManager = networkManager;
        this.metricsCollector = metricsCollector;
        this.storage = new ConcurrentHashMap<>();
        this.metrics = new HashMap<>();
        
        this.isRunning = false;
        this.isHealthy = true;
        
        logger.info("StorageService initialized: {}", serviceId);
    }
    
    @Override
    public String getServiceId() {
        return serviceId;
    }
    
    @Override
    public String getServiceType() {
        return "STORAGE_SERVICE";
    }
    
    @Override
    public String getLocation() {
        return "CLOUD_DATACENTER";
    }
    
    @Override
    public void start() {
        if (isRunning) {
            logger.warn("StorageService {} is already running", serviceId);
            return;
        }
        
        logger.info("Starting StorageService: {}", serviceId);
        isRunning = true;
        isHealthy = true;
        logger.info("StorageService {} started successfully", serviceId);
    }
    
    @Override
    public void stop() {
        if (!isRunning) {
            logger.warn("StorageService {} is not running", serviceId);
            return;
        }
        
        logger.info("Stopping StorageService: {}", serviceId);
        isRunning = false;
        logger.info("StorageService {} stopped successfully", serviceId);
    }
    
    @Override
    public boolean isRunning() {
        return isRunning;
    }
    
    @Override
    public boolean isHealthy() {
        return isHealthy && isRunning;
    }
    
    @Override
    public String processTask(String task) {
        if (!isRunning) {
            logger.warn("StorageService {} is not running, cannot process task", serviceId);
            return null;
        }
        
        try {
            logger.info("Processing storage task: {}", task);
            
            // Simulate task processing
            String result = "Storage task processed: " + task;
            lastTaskProcessingTime = System.currentTimeMillis();
            taskProcessingRate = 1.0; // Simulate processing rate
            
            logger.info("Storage task processed successfully: {}", task);
            return result;
            
        } catch (Exception e) {
            logger.error("Error processing storage task: {}", task, e);
            errorCount++;
            return null;
        }
    }
    
    @Override
    public boolean storeData(String data) {
        if (!isRunning) {
            logger.warn("StorageService {} is not running, cannot store data", serviceId);
            return false;
        }
        
        try {
            String dataId = "data_" + System.currentTimeMillis();
            storage.put(dataId, data);
            lastDataStorageTime = System.currentTimeMillis();
            dataStorageRate = 1.0; // Simulate storage rate
            
            logger.info("Data stored successfully with ID: {}", dataId);
            return true;
            
        } catch (Exception e) {
            logger.error("Error storing data", e);
            errorCount++;
            return false;
        }
    }
    
    @Override
    public String retrieveData(String dataId) {
        if (!isRunning) {
            logger.warn("StorageService {} is not running, cannot retrieve data", serviceId);
            return null;
        }
        
        try {
            String data = storage.get(dataId);
            if (data != null) {
                logger.info("Data retrieved successfully for ID: {}", dataId);
                return data;
            } else {
                logger.warn("Data not found for ID: {}", dataId);
                return null;
            }
            
        } catch (Exception e) {
            logger.error("Error retrieving data for ID: {}", dataId, e);
            errorCount++;
            return null;
        }
    }
    
    @Override
    public String getStatus() {
        return isRunning ? (isHealthy ? "ACTIVE" : "UNHEALTHY") : "STOPPED";
    }
    
    @Override
    public PerformanceMetrics getMetrics() {
        updateMetrics();
        PerformanceMetrics perfMetrics = new PerformanceMetrics(serviceId, "STORAGE_SERVICE");
        
        perfMetrics.addMetric("cpuUsage", cpuUsage);
        perfMetrics.addMetric("memoryUsage", memoryUsage);
        perfMetrics.addMetric("storageUsage", storageUsage);
        perfMetrics.addMetric("bandwidthUsage", bandwidthUsage);
        perfMetrics.addMetric("energyConsumption", energyConsumption);
        perfMetrics.addMetric("taskProcessingRate", taskProcessingRate);
        perfMetrics.addMetric("dataStorageRate", dataStorageRate);
        perfMetrics.addMetric("errorCount", errorCount);
        perfMetrics.addMetric("lastTaskProcessingTime", lastTaskProcessingTime);
        perfMetrics.addMetric("lastDataStorageTime", lastDataStorageTime);
        perfMetrics.addMetric("isHealthy", isHealthy());
        perfMetrics.addMetric("isRunning", isRunning());
        
        return perfMetrics;
    }
    
    @Override
    public void updateConfiguration(Map<String, Object> config) {
        logger.info("Updating StorageService configuration: {}", config);
        // Apply configuration updates
    }
    
    @Override
    public double getProcessingCapacity() {
        return 1000.0; // MB/s
    }
    
    @Override
    public double getStorageCapacity() {
        return 100000.0; // MB
    }
    
    @Override
    public double getCpuUsage() {
        return cpuUsage;
    }
    
    @Override
    public double getMemoryUsage() {
        return memoryUsage;
    }
    
    @Override
    public double getStorageUsage() {
        return storageUsage;
    }
    
    @Override
    public double getBandwidthUsage() {
        return bandwidthUsage;
    }
    
    @Override
    public double getEnergyConsumption() {
        return energyConsumption;
    }
    
    @Override
    public double getTaskProcessingRate() {
        return taskProcessingRate;
    }
    
    @Override
    public double getDataStorageRate() {
        return dataStorageRate;
    }
    
    @Override
    public int getErrorCount() {
        return errorCount;
    }
    
    @Override
    public void resetErrorCount() {
        errorCount = 0;
        logger.info("Error count reset for StorageService: {}", serviceId);
    }
    
    @Override
    public long getLastTaskProcessingTime() {
        return lastTaskProcessingTime;
    }
    
    @Override
    public long getLastDataStorageTime() {
        return lastDataStorageTime;
    }
    
    /**
     * Update service metrics
     */
    private void updateMetrics() {
        // Simulate resource usage
        cpuUsage = 20.0 + Math.random() * 30.0;
        memoryUsage = 40.0 + Math.random() * 20.0;
        storageUsage = (double) storage.size() / 1000.0 * 100.0; // Based on storage entries
        bandwidthUsage = 50.0 + Math.random() * 30.0;
        energyConsumption = 100.0 + Math.random() * 50.0;
        
        // Update metrics map
        metrics.put("cpu_usage", cpuUsage);
        metrics.put("memory_usage", memoryUsage);
        metrics.put("storage_usage", storageUsage);
        metrics.put("bandwidth_usage", bandwidthUsage);
        metrics.put("energy_consumption", energyConsumption);
        metrics.put("task_processing_rate", taskProcessingRate);
        metrics.put("data_storage_rate", dataStorageRate);
        metrics.put("error_count", errorCount);
        metrics.put("healthy", isHealthy);
        metrics.put("running", isRunning);
        metrics.put("last_task_processing_time", lastTaskProcessingTime);
        metrics.put("last_data_storage_time", lastDataStorageTime);
    }
    
    /**
     * Get stored data count
     * 
     * @return Number of stored data items
     */
    public int getStoredDataCount() {
        return storage.size();
    }
    
    /**
     * Clear all stored data
     */
    public void clearStorage() {
        storage.clear();
        logger.info("Storage cleared for StorageService: {}", serviceId);
    }
    
    @Override
    public String toString() {
        return String.format("StorageService{serviceId='%s', running=%s, healthy=%s, storedData=%d}",
            serviceId, isRunning, isHealthy, storage.size());
    }
} 