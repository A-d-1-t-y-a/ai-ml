package com.nci.fogedge.iot;

import com.nci.fogedge.network.NetworkManager;
import com.nci.fogedge.utils.MetricsCollector;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Base abstract class for IoT devices in the Fog and Edge Computing System
 * 
 * This class provides common functionality for all IoT devices including
 * lifecycle management, data transmission, performance tracking, and
 * LoRaWAN connectivity simulation.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public abstract class BaseIoTDevice implements IoTDevice {
    
    protected static final Logger logger = LoggerFactory.getLogger(BaseIoTDevice.class);
    
    // Device properties
    protected final String deviceId;
    protected final String deviceType;
    protected volatile String status;
    protected volatile boolean isRunning;
    
    // Dependencies
    protected final NetworkManager networkManager;
    protected final MetricsCollector metricsCollector;
    
    // Performance tracking
    protected final AtomicLong totalDataGenerated;
    protected final AtomicInteger successfulTransmissions;
    protected final AtomicInteger failedTransmissions;
    protected final AtomicInteger transmissionRate;
    
    // Device state
    protected volatile double batteryLevel;
    protected volatile double signalStrength;
    protected volatile long lastTransmissionTime;
    
    // Configuration
    protected final Map<String, Object> configuration;
    
    // Thread management
    protected ScheduledExecutorService deviceExecutor;
    protected ScheduledFuture<?> dataGenerationTask;
    protected ScheduledFuture<?> healthCheckTask;
    
    /**
     * Constructor for base IoT device
     * 
     * @param deviceId Unique device identifier
     * @param deviceType Type of device (e.g., "TEMPERATURE_SENSOR")
     * @param networkManager Network manager for communication
     * @param metricsCollector Metrics collector for performance tracking
     */
    protected BaseIoTDevice(String deviceId, String deviceType, 
                           NetworkManager networkManager, 
                           MetricsCollector metricsCollector) {
        this.deviceId = deviceId;
        this.deviceType = deviceType;
        this.networkManager = networkManager;
        this.metricsCollector = metricsCollector;
        
        this.status = "INACTIVE";
        this.isRunning = false;
        
        this.totalDataGenerated = new AtomicLong(0);
        this.successfulTransmissions = new AtomicInteger(0);
        this.failedTransmissions = new AtomicInteger(0);
        this.transmissionRate = new AtomicInteger(0);
        
        this.batteryLevel = 100.0; // Start with full battery
        this.signalStrength = -50.0; // Good signal strength
        this.lastTransmissionTime = System.currentTimeMillis();
        
        this.configuration = new ConcurrentHashMap<>();
        initializeDefaultConfiguration();
        
        logger.debug("Base IoT device initialized: {}", deviceId);
    }
    
    /**
     * Initialize default configuration for the device
     */
    protected void initializeDefaultConfiguration() {
        configuration.put("dataGenerationInterval", 30); // seconds
        configuration.put("transmissionPower", 14); // dBm
        configuration.put("batteryConsumption", 0.1); // % per transmission
        configuration.put("signalVariation", 5.0); // dBm
        configuration.put("maxRetries", 3);
        configuration.put("timeout", 5000); // milliseconds
    }
    
    @Override
    public String getDeviceId() {
        return deviceId;
    }
    
    @Override
    public String getDeviceType() {
        return deviceType;
    }
    
    @Override
    public String getStatus() {
        return status;
    }
    
    @Override
    public boolean isHealthy() {
        return isRunning && batteryLevel > 10.0 && signalStrength > -80.0;
    }
    
    @Override
    public void start() {
        if (isRunning) {
            logger.warn("Device {} is already running", deviceId);
            return;
        }
        
        logger.info("Starting device: {}", deviceId);
        
        try {
            // Initialize device-specific components
            initializeDevice();
            
            // Create executor for device tasks
            deviceExecutor = Executors.newScheduledThreadPool(2);
            
            // Start data generation task
            int interval = (Integer) configuration.get("dataGenerationInterval");
            dataGenerationTask = deviceExecutor.scheduleAtFixedRate(() -> {
                try {
                    generateAndTransmitData();
                } catch (Exception e) {
                    logger.error("Error in data generation for device: {}", deviceId, e);
                }
            }, 0, interval, TimeUnit.SECONDS);
            
            // Start health check task
            healthCheckTask = deviceExecutor.scheduleAtFixedRate(() -> {
                try {
                    performHealthCheck();
                } catch (Exception e) {
                    logger.error("Error in health check for device: {}", deviceId, e);
                }
            }, 10, 60, TimeUnit.SECONDS);
            
            isRunning = true;
            status = "ACTIVE";
            
            logger.info("Device {} started successfully", deviceId);
            
        } catch (Exception e) {
            logger.error("Failed to start device: {}", deviceId, e);
            status = "ERROR";
            throw new RuntimeException("Device startup failed", e);
        }
    }
    
    @Override
    public void stop() {
        if (!isRunning) {
            logger.warn("Device {} is not running", deviceId);
            return;
        }
        
        logger.info("Stopping device: {}", deviceId);
        
        try {
            // Stop scheduled tasks
            if (dataGenerationTask != null) {
                dataGenerationTask.cancel(true);
            }
            if (healthCheckTask != null) {
                healthCheckTask.cancel(true);
            }
            
            // Shutdown executor
            if (deviceExecutor != null) {
                deviceExecutor.shutdown();
                if (!deviceExecutor.awaitTermination(10, TimeUnit.SECONDS)) {
                    deviceExecutor.shutdownNow();
                }
            }
            
            // Perform device-specific cleanup
            cleanupDevice();
            
            isRunning = false;
            status = "INACTIVE";
            
            logger.info("Device {} stopped successfully", deviceId);
            
        } catch (Exception e) {
            logger.error("Error stopping device: {}", deviceId, e);
            status = "ERROR";
        }
    }
    
    @Override
    public double getBatteryLevel() {
        return batteryLevel;
    }
    
    @Override
    public double getSignalStrength() {
        return signalStrength;
    }
    
    @Override
    public double getTransmissionRate() {
        return transmissionRate.get();
    }
    
    @Override
    public long getTotalDataGenerated() {
        return totalDataGenerated.get();
    }
    
    @Override
    public int getSuccessfulTransmissions() {
        return successfulTransmissions.get();
    }
    
    @Override
    public int getFailedTransmissions() {
        return failedTransmissions.get();
    }
    
    @Override
    public double getTransmissionSuccessRate() {
        int total = successfulTransmissions.get() + failedTransmissions.get();
        return total > 0 ? (double) successfulTransmissions.get() / total * 100 : 0;
    }
    
    @Override
    public Map<String, Object> getConfiguration() {
        return new HashMap<>(configuration);
    }
    
    @Override
    public void updateConfiguration(Map<String, Object> config) {
        configuration.putAll(config);
        logger.info("Configuration updated for device: {}", deviceId);
    }
    
    @Override
    public Map<String, Object> getPerformanceMetrics() {
        Map<String, Object> metrics = new HashMap<>();
        metrics.put("deviceId", deviceId);
        metrics.put("deviceType", deviceType);
        metrics.put("status", status);
        metrics.put("batteryLevel", batteryLevel);
        metrics.put("signalStrength", signalStrength);
        metrics.put("totalDataGenerated", totalDataGenerated.get());
        metrics.put("successfulTransmissions", successfulTransmissions.get());
        metrics.put("failedTransmissions", failedTransmissions.get());
        metrics.put("transmissionSuccessRate", getTransmissionSuccessRate());
        metrics.put("transmissionRate", transmissionRate.get());
        metrics.put("lastTransmissionTime", lastTransmissionTime);
        
        return metrics;
    }
    
    @Override
    public void resetStatistics() {
        totalDataGenerated.set(0);
        successfulTransmissions.set(0);
        failedTransmissions.set(0);
        transmissionRate.set(0);
        lastTransmissionTime = System.currentTimeMillis();
        
        logger.info("Statistics reset for device: {}", deviceId);
    }
    
    @Override
    public DiagnosticResult performDiagnostic() {
        logger.debug("Performing diagnostic for device: {}", deviceId);
        
        Map<String, Object> details = new HashMap<>();
        boolean passed = true;
        String message = "Diagnostic passed";
        
        // Check battery level
        if (batteryLevel < 20.0) {
            passed = false;
            message = "Low battery level";
        }
        details.put("batteryLevel", batteryLevel);
        
        // Check signal strength
        if (signalStrength < -80.0) {
            passed = false;
            message = "Poor signal strength";
        }
        details.put("signalStrength", signalStrength);
        
        // Check transmission success rate
        double successRate = getTransmissionSuccessRate();
        if (successRate < 80.0) {
            passed = false;
            message = "Low transmission success rate";
        }
        details.put("transmissionSuccessRate", successRate);
        
        // Check device status
        details.put("status", status);
        details.put("isRunning", isRunning);
        
        return new DiagnosticResult(passed, message, details);
    }
    
    /**
     * Generate and transmit data to the network
     */
    protected void generateAndTransmitData() {
        try {
            // Generate device-specific data
            Object data = generateData();
            
            if (data != null) {
                // Simulate data transmission via LoRaWAN
                boolean transmissionSuccess = transmitData(data);
                
                if (transmissionSuccess) {
                    successfulTransmissions.incrementAndGet();
                    logger.debug("Data transmitted successfully from device: {}", deviceId);
                } else {
                    failedTransmissions.incrementAndGet();
                    logger.warn("Data transmission failed from device: {}", deviceId);
                }
                
                // Update transmission rate
                long currentTime = System.currentTimeMillis();
                long timeDiff = currentTime - lastTransmissionTime;
                if (timeDiff > 0) {
                    int rate = (int) (1000.0 / timeDiff); // transmissions per second
                    transmissionRate.set(rate);
                }
                lastTransmissionTime = currentTime;
                
                // Consume battery
                consumeBattery();
                
                // Update signal strength (simulate variation)
                updateSignalStrength();
            }
            
        } catch (Exception e) {
            logger.error("Error in data generation and transmission for device: {}", deviceId, e);
            failedTransmissions.incrementAndGet();
        }
    }
    
    /**
     * Transmit data to the network layer
     * 
     * @param data Data to transmit
     * @return true if transmission successful, false otherwise
     */
    protected boolean transmitData(Object data) {
        try {
            // Simulate LoRaWAN transmission with network conditions
            double transmissionSuccessProbability = calculateTransmissionSuccessProbability();
            
            if (Math.random() < transmissionSuccessProbability) {
                // Simulate network transmission
                networkManager.transmitData(deviceId, data.toString().getBytes());
                return true;
            } else {
                return false;
            }
            
        } catch (Exception e) {
            logger.error("Error transmitting data from device: {}", deviceId, e);
            return false;
        }
    }
    
    /**
     * Calculate transmission success probability based on signal strength and battery
     * 
     * @return Probability of successful transmission (0.0 to 1.0)
     */
    protected double calculateTransmissionSuccessProbability() {
        // Base probability from signal strength
        double signalProbability = Math.max(0.0, (signalStrength + 100.0) / 100.0);
        
        // Battery factor
        double batteryFactor = batteryLevel / 100.0;
        
        // Combine factors
        return signalProbability * batteryFactor * 0.95; // 95% base success rate
    }
    
    /**
     * Consume battery during transmission
     */
    protected void consumeBattery() {
        double consumption = (Double) configuration.get("batteryConsumption");
        batteryLevel = Math.max(0.0, batteryLevel - consumption);
    }
    
    /**
     * Update signal strength with random variation
     */
    protected void updateSignalStrength() {
        double variation = (Double) configuration.get("signalVariation");
        double change = (Math.random() - 0.5) * variation;
        signalStrength = Math.max(-100.0, Math.min(-30.0, signalStrength + change));
    }
    
    /**
     * Perform periodic health check
     */
    protected void performHealthCheck() {
        DiagnosticResult result = performDiagnostic();
        
        if (!result.isPassed()) {
            logger.warn("Health check failed for device {}: {}", deviceId, result.getMessage());
            status = "WARNING";
        } else {
            status = "ACTIVE";
        }
        
        // Update metrics
        Map<String, Object> metrics = getPerformanceMetrics();
        PerformanceMetrics perfMetrics = new PerformanceMetrics(
            deviceId, "DEVICE", 
            (Double) metrics.get("latency"), 
            (Double) metrics.get("throughput"),
            (Double) metrics.get("energyConsumption"),
            (Double) metrics.get("cpuUsage"),
            (Double) metrics.get("memoryUsage"),
            (Long) metrics.get("dataProcessed"),
            (Integer) metrics.get("activeConnections"),
            (Double) metrics.get("healthScore")
        );
        metricsCollector.updateDeviceMetrics(deviceId, perfMetrics);
    }
    
    /**
     * Initialize device-specific components
     */
    protected abstract void initializeDevice();
    
    /**
     * Cleanup device-specific resources
     */
    protected abstract void cleanupDevice();
} 