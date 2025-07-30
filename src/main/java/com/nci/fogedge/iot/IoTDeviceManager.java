package com.nci.fogedge.iot;

import com.nci.fogedge.network.NetworkManager;
import com.nci.fogedge.utils.MetricsCollector;
import com.nci.fogedge.utils.ConfigurationManager;
import com.nci.fogedge.iot.devices.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * IoT Device Manager for the Fog and Edge Computing System
 * 
 * This class manages multiple IoT devices including sensors and actuators,
 * implementing LoRaWAN connectivity for wireless communication with edge nodes.
 * Based on the research paper's IoT layer implementation.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class IoTDeviceManager {
    
    private static final Logger logger = LoggerFactory.getLogger(IoTDeviceManager.class);
    
    // Configuration and dependencies
    private final ConfigurationManager configManager;
    private final NetworkManager networkManager;
    private final MetricsCollector metricsCollector;
    
    // Device management
    private final Map<String, IoTDevice> devices;
    private final List<IoTDevice> activeDevices;
    private final AtomicInteger deviceCounter;
    
    // Thread management
    private final ScheduledExecutorService deviceExecutor;
    private final List<Future<?>> deviceTasks;
    
    // Performance tracking
    private final AtomicInteger totalDataGenerated;
    private final AtomicInteger successfulTransmissions;
    private final AtomicInteger failedTransmissions;
    
    /**
     * Constructor for IoT Device Manager
     * 
     * @param configManager Configuration manager for system settings
     * @param networkManager Network manager for communication
     * @param metricsCollector Metrics collector for performance tracking
     */
    public IoTDeviceManager(ConfigurationManager configManager, 
                           NetworkManager networkManager, 
                           MetricsCollector metricsCollector) {
        this.configManager = configManager;
        this.networkManager = networkManager;
        this.metricsCollector = metricsCollector;
        
        this.devices = new ConcurrentHashMap<>();
        this.activeDevices = Collections.synchronizedList(new ArrayList<>());
        this.deviceCounter = new AtomicInteger(0);
        
        this.deviceExecutor = Executors.newScheduledThreadPool(20);
        this.deviceTasks = Collections.synchronizedList(new ArrayList<>());
        
        this.totalDataGenerated = new AtomicInteger(0);
        this.successfulTransmissions = new AtomicInteger(0);
        this.failedTransmissions = new AtomicInteger(0);
        
        logger.info("IoT Device Manager initialized");
    }
    
    /**
     * Start the IoT device manager and initialize devices
     */
    public void start() {
        logger.info("Starting IoT Device Manager...");
        
        try {
            // Create and initialize IoT devices
            createIoTDevices();
            
            // Start all devices
            startAllDevices();
            
            // Start device monitoring
            startDeviceMonitoring();
            
            logger.info("IoT Device Manager started successfully with {} devices", activeDevices.size());
            
        } catch (Exception e) {
            logger.error("Failed to start IoT Device Manager", e);
            throw new RuntimeException("IoT Device Manager startup failed", e);
        }
    }
    
    /**
     * Create various types of IoT devices
     */
    private void createIoTDevices() {
        logger.info("Creating IoT devices...");
        
        // Temperature sensors
        for (int i = 0; i < 10; i++) {
            String deviceId = "TEMP_" + String.format("%03d", i);
            TemperatureSensor sensor = new TemperatureSensor(deviceId, networkManager, metricsCollector);
            devices.put(deviceId, sensor);
            activeDevices.add(sensor);
            logger.debug("Created temperature sensor: {}", deviceId);
        }
        
        // Humidity sensors
        for (int i = 0; i < 8; i++) {
            String deviceId = "HUMID_" + String.format("%03d", i);
            HumiditySensor sensor = new HumiditySensor(deviceId, networkManager, metricsCollector);
            devices.put(deviceId, sensor);
            activeDevices.add(sensor);
            logger.debug("Created humidity sensor: {}", deviceId);
        }
        
        // Pressure sensors
        for (int i = 0; i < 6; i++) {
            String deviceId = "PRESSURE_" + String.format("%03d", i);
            PressureSensor sensor = new PressureSensor(deviceId, networkManager, metricsCollector);
            devices.put(deviceId, sensor);
            activeDevices.add(sensor);
            logger.debug("Created pressure sensor: {}", deviceId);
        }
        
        // Light sensors
        for (int i = 0; i < 5; i++) {
            String deviceId = "LIGHT_" + String.format("%03d", i);
            LightSensor sensor = new LightSensor(deviceId, networkManager, metricsCollector);
            devices.put(deviceId, sensor);
            activeDevices.add(sensor);
            logger.debug("Created light sensor: {}", deviceId);
        }
        
        // Motion sensors
        for (int i = 0; i < 4; i++) {
            String deviceId = "MOTION_" + String.format("%03d", i);
            MotionSensor sensor = new MotionSensor(deviceId, networkManager, metricsCollector);
            devices.put(deviceId, sensor);
            activeDevices.add(sensor);
            logger.debug("Created motion sensor: {}", deviceId);
        }
        
        // Actuators
        for (int i = 0; i < 3; i++) {
            String deviceId = "ACTUATOR_" + String.format("%03d", i);
            SmartActuator actuator = new SmartActuator(deviceId, networkManager, metricsCollector);
            devices.put(deviceId, actuator);
            activeDevices.add(actuator);
            logger.debug("Created smart actuator: {}", deviceId);
        }
        
        logger.info("Created {} IoT devices successfully", activeDevices.size());
    }
    
    /**
     * Start all IoT devices
     */
    private void startAllDevices() {
        logger.info("Starting all IoT devices...");
        
        for (IoTDevice device : activeDevices) {
            try {
                device.start();
                logger.debug("Started device: {}", device.getDeviceId());
            } catch (Exception e) {
                logger.error("Failed to start device: {}", device.getDeviceId(), e);
            }
        }
        
        logger.info("All IoT devices started");
    }
    
    /**
     * Start device monitoring and data collection
     */
    private void startDeviceMonitoring() {
        logger.info("Starting device monitoring...");
        
        // Monitor device health
        Future<?> healthMonitor = deviceExecutor.scheduleAtFixedRate(() -> {
            try {
                monitorDeviceHealth();
            } catch (Exception e) {
                logger.error("Error in device health monitoring", e);
            }
        }, 10, 60, TimeUnit.SECONDS);
        deviceTasks.add(healthMonitor);
        
        // Monitor data transmission
        Future<?> transmissionMonitor = deviceExecutor.scheduleAtFixedRate(() -> {
            try {
                monitorDataTransmission();
            } catch (Exception e) {
                logger.error("Error in data transmission monitoring", e);
            }
        }, 15, 45, TimeUnit.SECONDS);
        deviceTasks.add(transmissionMonitor);
        
        logger.info("Device monitoring started");
    }
    
    /**
     * Monitor the health of all IoT devices
     */
    private void monitorDeviceHealth() {
        logger.debug("Monitoring device health...");
        
        int healthyDevices = 0;
        int totalDevices = activeDevices.size();
        
        for (IoTDevice device : activeDevices) {
            if (device.isHealthy()) {
                healthyDevices++;
            } else {
                logger.warn("Device {} is unhealthy", device.getDeviceId());
            }
        }
        
        double healthPercentage = (double) healthyDevices / totalDevices * 100;
        logger.info("Device Health Status: {}/{} devices healthy ({:.2f}%)", 
                   healthyDevices, totalDevices, healthPercentage);
        
        // Update metrics
        metricsCollector.updateDeviceHealth(healthPercentage);
    }
    
    /**
     * Monitor data transmission statistics
     */
    private void monitorDataTransmission() {
        logger.debug("Monitoring data transmission...");
        
        int totalData = totalDataGenerated.get();
        int successful = successfulTransmissions.get();
        int failed = failedTransmissions.get();
        
        double successRate = totalData > 0 ? (double) successful / totalData * 100 : 0;
        
        logger.info("Data Transmission Stats:");
        logger.info("  Total Data Generated: {} bytes", totalData);
        logger.info("  Successful Transmissions: {}", successful);
        logger.info("  Failed Transmissions: {}", failed);
        logger.info("  Success Rate: {:.2f}%", successRate);
        
        // Update metrics
        metricsCollector.updateTransmissionStats(totalData, successful, failed, successRate);
    }
    
    /**
     * Get the count of active devices
     * 
     * @return Number of active devices
     */
    public int getActiveDeviceCount() {
        return activeDevices.size();
    }
    
    /**
     * Get a specific device by ID
     * 
     * @param deviceId Device identifier
     * @return IoT device or null if not found
     */
    public IoTDevice getDevice(String deviceId) {
        return devices.get(deviceId);
    }
    
    /**
     * Get all active devices
     * 
     * @return List of active devices
     */
    public List<IoTDevice> getAllDevices() {
        return new ArrayList<>(activeDevices);
    }
    
    /**
     * Record successful data transmission
     */
    public void recordSuccessfulTransmission() {
        successfulTransmissions.incrementAndGet();
    }
    
    /**
     * Record failed data transmission
     */
    public void recordFailedTransmission() {
        failedTransmissions.incrementAndGet();
    }
    
    /**
     * Record data generation
     * 
     * @param dataSize Size of generated data in bytes
     */
    public void recordDataGeneration(int dataSize) {
        totalDataGenerated.addAndGet(dataSize);
    }
    
    /**
     * Stop the IoT device manager
     */
    public void stop() {
        logger.info("Stopping IoT Device Manager...");
        
        try {
            // Stop all devices
            for (IoTDevice device : activeDevices) {
                try {
                    device.stop();
                    logger.debug("Stopped device: {}", device.getDeviceId());
                } catch (Exception e) {
                    logger.error("Error stopping device: {}", device.getDeviceId(), e);
                }
            }
            
            // Cancel all monitoring tasks
            for (Future<?> task : deviceTasks) {
                if (!task.isCancelled()) {
                    task.cancel(true);
                }
            }
            
            // Shutdown executor
            deviceExecutor.shutdown();
            if (!deviceExecutor.awaitTermination(30, TimeUnit.SECONDS)) {
                deviceExecutor.shutdownNow();
            }
            
            logger.info("IoT Device Manager stopped successfully");
            
        } catch (Exception e) {
            logger.error("Error stopping IoT Device Manager", e);
        }
    }
    
    /**
     * Get performance statistics
     * 
     * @return Map containing performance statistics
     */
    public Map<String, Object> getPerformanceStats() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("totalDevices", activeDevices.size());
        stats.put("totalDataGenerated", totalDataGenerated.get());
        stats.put("successfulTransmissions", successfulTransmissions.get());
        stats.put("failedTransmissions", failedTransmissions.get());
        stats.put("successRate", totalDataGenerated.get() > 0 ? 
                  (double) successfulTransmissions.get() / totalDataGenerated.get() * 100 : 0);
        
        return stats;
    }
} 