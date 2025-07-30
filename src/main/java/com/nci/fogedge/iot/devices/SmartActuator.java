package com.nci.fogedge.iot.devices;

import com.nci.fogedge.iot.BaseIoTDevice;
import com.nci.fogedge.iot.IoTDevice;
import com.nci.fogedge.network.NetworkManager;
import com.nci.fogedge.utils.MetricsCollector;
import com.nci.fogedge.utils.DiagnosticResult;
import com.nci.fogedge.utils.PerformanceMetrics;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Smart Actuator implementation for the Fog and Edge Computing System
 * 
 * This class simulates a smart actuator that receives control commands from
 * edge nodes and performs physical actions. Based on the research paper's
 * IoT actuator implementation.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class SmartActuator extends BaseIoTDevice {
    
    private static final Logger logger = LoggerFactory.getLogger(SmartActuator.class);
    
    // Actuator specific properties
    private String currentState;
    private String actuatorType;
    private int operationCount;
    private long lastOperationTime;
    private Random random;
    
    // Actuator configuration
    private double powerConsumption;
    private double responseTime;
    private boolean isOperational;
    
    /**
     * Constructor for Smart Actuator
     * 
     * @param deviceId Unique device identifier
     * @param networkManager Network manager for communication
     * @param metricsCollector Metrics collector for performance tracking
     */
    public SmartActuator(String deviceId, NetworkManager networkManager, MetricsCollector metricsCollector) {
        super(deviceId, "SMART_ACTUATOR", networkManager, metricsCollector);
        
        this.random = new Random();
        this.currentState = "IDLE";
        this.actuatorType = "GENERIC";
        this.operationCount = 0;
        this.lastOperationTime = 0;
        this.powerConsumption = 10.0; // 10W power consumption
        this.responseTime = 0.5; // 0.5 seconds response time
        this.isOperational = true;
        
        logger.debug("Smart actuator initialized: {} with type: {}", deviceId, actuatorType);
    }
    
    @Override
    protected void initializeDevice() {
        logger.debug("Initializing smart actuator: {}", deviceId);
        
        // Set device-specific configuration
        configuration.put("actuatorType", actuatorType);
        configuration.put("powerConsumption", powerConsumption);
        configuration.put("responseTime", responseTime);
        configuration.put("maxOperations", 10000);
        configuration.put("operationalMode", "AUTO");
        
        logger.debug("Smart actuator {} initialized successfully", deviceId);
    }
    
    @Override
    protected void cleanupDevice() {
        logger.debug("Cleaning up smart actuator: {}", deviceId);
        
        // Save operation statistics
        saveOperationStats();
        
        logger.debug("Smart actuator {} cleanup completed", deviceId);
    }
    
    @Override
    public String collectData() {
        try {
            long currentTime = System.currentTimeMillis();
            
            // Simulate actuator status monitoring
            if (random.nextDouble() < 0.05) { // 5% chance of state change
                performRandomOperation();
            }
            
            // Create actuator status data JSON string
            String actuatorData = String.format(
                "{\"deviceId\":\"%s\",\"deviceType\":\"SMART_ACTUATOR\",\"timestamp\":%d,\"currentState\":\"%s\",\"actuatorType\":\"%s\",\"operationCount\":%d,\"lastOperationTime\":%d,\"powerConsumption\":%.1f,\"responseTime\":%.1f,\"isOperational\":%s,\"batteryLevel\":%.1f,\"signalStrength\":%.1f,\"status\":\"%s\"}",
                deviceId,
                currentTime,
                currentState,
                actuatorType,
                operationCount,
                lastOperationTime,
                powerConsumption,
                responseTime,
                isOperational,
                batteryLevel,
                signalStrength,
                status
            );
            
            // Update total data generated
            totalDataGenerated.addAndGet(actuatorData.getBytes().length);
            
            return actuatorData;
            
        } catch (Exception e) {
            logger.error("Error generating actuator data for device: {}", deviceId, e);
            return null;
        }
    }
    
    @Override
    public boolean transmitData(String data) {
        try {
            if (data != null && !data.isEmpty()) {
                boolean success = networkManager.transmitData(deviceId, data);
                if (success) {
                    successfulTransmissions.incrementAndGet();
                    logger.debug("Smart actuator {} transmitted data successfully", deviceId);
                } else {
                    failedTransmissions.incrementAndGet();
                    logger.warn("Smart actuator {} failed to transmit data", deviceId);
                }
                return success;
            }
            return false;
        } catch (Exception e) {
            failedTransmissions.incrementAndGet();
            logger.error("Error transmitting data from smart actuator: {}", deviceId, e);
            return false;
        }
    }
    
    @Override
    public PerformanceMetrics getMetrics() {
        PerformanceMetrics metrics = new PerformanceMetrics(deviceId, "SMART_ACTUATOR");
        
        metrics.addMetric("batteryLevel", batteryLevel);
        metrics.addMetric("signalStrength", signalStrength);
        metrics.addMetric("transmissionRate", getTransmissionRate());
        metrics.addMetric("errorCount", getErrorCount());
        metrics.addMetric("currentState", currentState);
        metrics.addMetric("actuatorType", actuatorType);
        metrics.addMetric("operationCount", operationCount);
        metrics.addMetric("lastOperationTime", lastOperationTime);
        metrics.addMetric("powerConsumption", powerConsumption);
        metrics.addMetric("responseTime", responseTime);
        metrics.addMetric("isOperational", isOperational);
        metrics.addMetric("isHealthy", isHealthy());
        metrics.addMetric("isRunning", isRunning());
        
        return metrics;
    }
    
    @Override
    public void updateConfiguration(Map<String, Object> config) {
        if (config != null) {
            configuration.putAll(config);
            logger.debug("Smart actuator {} configuration updated", deviceId);
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
        long total = successfulTransmissions.get() + failedTransmissions.get();
        return total > 0 ? (double) successfulTransmissions.get() / total : 0.0;
    }
    
    @Override
    public int getErrorCount() {
        return failedTransmissions.get();
    }
    
    @Override
    public void resetErrorCount() {
        failedTransmissions.set(0);
    }
    
    @Override
    public long getLastDataCollectionTime() {
        return System.currentTimeMillis();
    }
    
    @Override
    public long getLastTransmissionTime() {
        return System.currentTimeMillis();
    }
    
    /**
     * Perform a random operation to simulate actuator activity
     */
    private void performRandomOperation() {
        String[] possibleStates = {"ON", "OFF", "IDLE", "ACTIVE", "STANDBY"};
        String newState = possibleStates[random.nextInt(possibleStates.length)];
        
        if (!newState.equals(currentState)) {
            currentState = newState;
            operationCount++;
            lastOperationTime = System.currentTimeMillis();
            
            logger.debug("Actuator {} changed state to: {}", deviceId, newState);
        }
    }
    
    /**
     * Execute a control command from edge node
     * 
     * @param command Control command to execute
     * @return true if command executed successfully, false otherwise
     */
    public boolean executeCommand(String command) {
        try {
            logger.info("Executing command '{}' on actuator: {}", command, deviceId);
            
            // Simulate command execution
            if (isOperational && batteryLevel > 20.0) {
                currentState = command.toUpperCase();
                operationCount++;
                lastOperationTime = System.currentTimeMillis();
                
                // Simulate power consumption
                batteryLevel = Math.max(0.0, batteryLevel - powerConsumption * 0.01);
                
                logger.info("Command '{}' executed successfully on actuator: {}", command, deviceId);
                return true;
            } else {
                logger.warn("Actuator {} cannot execute command: not operational or low battery", deviceId);
                return false;
            }
            
        } catch (Exception e) {
            logger.error("Error executing command on actuator: {}", deviceId, e);
            return false;
        }
    }
    
    /**
     * Save operation statistics
     */
    private void saveOperationStats() {
        // In a real implementation, this would save to persistent storage
        logger.debug("Operation statistics saved for actuator: {}", deviceId);
    }
    
    /**
     * Get current actuator state
     * 
     * @return Current state of the actuator
     */
    public String getCurrentState() {
        return currentState;
    }
    
    /**
     * Get actuator type
     * 
     * @return Type of the actuator
     */
    public String getActuatorType() {
        return actuatorType;
    }
    
    /**
     * Get total operation count
     * 
     * @return Total number of operations performed
     */
    public int getOperationCount() {
        return operationCount;
    }
    
    /**
     * Get last operation time
     * 
     * @return Timestamp of last operation
     */
    public long getLastOperationTime() {
        return lastOperationTime;
    }
    
    /**
     * Get power consumption
     * 
     * @return Power consumption in watts
     */
    public double getPowerConsumption() {
        return powerConsumption;
    }
    
    /**
     * Get response time
     * 
     * @return Response time in seconds
     */
    public double getResponseTime() {
        return responseTime;
    }
    
    /**
     * Check if actuator is operational
     * 
     * @return true if operational, false otherwise
     */
    public boolean isOperational() {
        return isOperational;
    }
    
    @Override
    public DiagnosticResult performDiagnostic() {
        Map<String, Object> details = new HashMap<>();
        boolean passed = true;
        String message = "Smart actuator diagnostic passed";
        
        // Add actuator-specific diagnostic checks
        if (!isOperational) {
            passed = false;
            message = "Actuator not operational";
        }
        details.put("isOperational", isOperational);
        
        if (operationCount > 10000) {
            passed = false;
            message = "Actuator exceeded maximum operations";
        }
        details.put("operationCount", operationCount);
        details.put("maxOperations", 10000);
        
        if (responseTime > 2.0) {
            passed = false;
            message = "Actuator response time too slow";
        }
        details.put("responseTime", responseTime);
        details.put("maxResponseTime", 2.0);
        
        details.put("currentState", currentState);
        details.put("actuatorType", actuatorType);
        details.put("batteryLevel", batteryLevel);
        details.put("signalStrength", signalStrength);
        details.put("status", status);
        
        return passed ? DiagnosticResult.success(message, details) : DiagnosticResult.failure(message, details);
    }
} 