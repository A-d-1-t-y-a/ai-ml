package com.nci.fogedge.iot.devices;

import com.nci.fogedge.iot.BaseIoTDevice;
import com.nci.fogedge.network.NetworkManager;
import com.nci.fogedge.utils.MetricsCollector;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Motion Sensor implementation for the Fog and Edge Computing System
 * 
 * This class simulates a motion sensor that detects movement and transmits
 * detection events via LoRaWAN to edge nodes.
 * Based on the research paper's IoT sensor implementation.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class MotionSensor extends BaseIoTDevice {
    
    private static final Logger logger = LoggerFactory.getLogger(MotionSensor.class);
    
    // Motion sensor specific properties
    private boolean motionDetected;
    private long lastMotionTime;
    private int detectionCount;
    private Random random;
    
    // Sensor configuration
    private double detectionRange;
    private double sensitivity;
    private long cooldownPeriod;
    
    /**
     * Constructor for Motion Sensor
     * 
     * @param deviceId Unique device identifier
     * @param networkManager Network manager for communication
     * @param metricsCollector Metrics collector for performance tracking
     */
    public MotionSensor(String deviceId, NetworkManager networkManager, MetricsCollector metricsCollector) {
        super(deviceId, "MOTION_SENSOR", networkManager, metricsCollector);
        
        this.random = new Random();
        this.motionDetected = false;
        this.lastMotionTime = 0;
        this.detectionCount = 0;
        this.detectionRange = 5.0; // 5 meters detection range
        this.sensitivity = 0.8; // 80% sensitivity
        this.cooldownPeriod = 30000; // 30 seconds cooldown
        
        logger.debug("Motion sensor initialized: {} with detection range: {}m", deviceId, detectionRange);
    }
    
    @Override
    protected void initializeDevice() {
        logger.debug("Initializing motion sensor: {}", deviceId);
        
        // Set device-specific configuration
        configuration.put("sensorType", "PIR");
        configuration.put("detectionRange", detectionRange);
        configuration.put("sensitivity", sensitivity);
        configuration.put("cooldownPeriod", cooldownPeriod);
        configuration.put("responseTime", "0.1"); // seconds
        configuration.put("samplingRate", "10.0"); // Hz
        
        logger.debug("Motion sensor {} initialized successfully", deviceId);
    }
    
    @Override
    protected void cleanupDevice() {
        logger.debug("Cleaning up motion sensor: {}", deviceId);
        
        // Save detection statistics
        saveDetectionStats();
        
        logger.debug("Motion sensor {} cleanup completed", deviceId);
    }
    
    @Override
    public Object generateData() {
        try {
            long currentTime = System.currentTimeMillis();
            
            // Check if enough time has passed since last detection
            if (currentTime - lastMotionTime > cooldownPeriod) {
                // Simulate motion detection with probability based on sensitivity
                if (random.nextDouble() < sensitivity * 0.1) { // 10% chance per reading
                    motionDetected = true;
                    lastMotionTime = currentTime;
                    detectionCount++;
                    
                    logger.debug("Motion detected by sensor: {}", deviceId);
                } else {
                    motionDetected = false;
                }
            } else {
                motionDetected = false;
            }
            
            // Create motion data object
            Map<String, Object> motionData = new HashMap<>();
            motionData.put("deviceId", deviceId);
            motionData.put("deviceType", "MOTION_SENSOR");
            motionData.put("timestamp", currentTime);
            motionData.put("motionDetected", motionDetected);
            motionData.put("detectionCount", detectionCount);
            motionData.put("lastMotionTime", lastMotionTime);
            motionData.put("detectionRange", detectionRange);
            motionData.put("sensitivity", sensitivity);
            motionData.put("batteryLevel", batteryLevel);
            motionData.put("signalStrength", signalStrength);
            motionData.put("status", status);
            
            // Update total data generated
            totalDataGenerated.addAndGet(motionData.toString().getBytes().length);
            
            return motionData;
            
        } catch (Exception e) {
            logger.error("Error generating motion data for device: {}", deviceId, e);
            return null;
        }
    }
    
    /**
     * Save detection statistics
     */
    private void saveDetectionStats() {
        // In a real implementation, this would save to persistent storage
        logger.debug("Detection statistics saved for motion sensor: {}", deviceId);
    }
    
    /**
     * Get current motion detection status
     * 
     * @return true if motion is currently detected, false otherwise
     */
    public boolean isMotionDetected() {
        return motionDetected;
    }
    
    /**
     * Get total detection count
     * 
     * @return Total number of motion detections
     */
    public int getDetectionCount() {
        return detectionCount;
    }
    
    /**
     * Get last motion detection time
     * 
     * @return Timestamp of last motion detection
     */
    public long getLastMotionTime() {
        return lastMotionTime;
    }
    
    /**
     * Get detection range
     * 
     * @return Detection range in meters
     */
    public double getDetectionRange() {
        return detectionRange;
    }
    
    /**
     * Get sensor sensitivity
     * 
     * @return Sensor sensitivity (0.0 to 1.0)
     */
    public double getSensitivity() {
        return sensitivity;
    }
    
    @Override
    public Map<String, Object> getPerformanceMetrics() {
        Map<String, Object> metrics = super.getPerformanceMetrics();
        
        // Add motion-specific metrics
        metrics.put("motionDetected", motionDetected);
        metrics.put("detectionCount", detectionCount);
        metrics.put("lastMotionTime", lastMotionTime);
        metrics.put("detectionRange", detectionRange);
        metrics.put("sensitivity", sensitivity);
        metrics.put("cooldownPeriod", cooldownPeriod);
        
        return metrics;
    }
    
    @Override
    public IoTDevice.DiagnosticResult performDiagnostic() {
        IoTDevice.DiagnosticResult baseResult = super.performDiagnostic();
        
        Map<String, Object> details = new HashMap<>(baseResult.getDetails());
        boolean passed = baseResult.isPassed();
        String message = baseResult.getMessage();
        
        // Add motion-specific diagnostic checks
        if (detectionRange < 1.0 || detectionRange > 20.0) {
            passed = false;
            message = "Detection range out of valid range";
        }
        details.put("detectionRange", detectionRange);
        details.put("detectionRangeValid", "1.0 to 20.0 meters");
        
        if (sensitivity < 0.1 || sensitivity > 1.0) {
            passed = false;
            message = "Sensitivity out of valid range";
        }
        details.put("sensitivity", sensitivity);
        details.put("sensitivityValid", "0.1 to 1.0");
        
        details.put("motionDetected", motionDetected);
        details.put("detectionCount", detectionCount);
        
        return new IoTDevice.DiagnosticResult(passed, message, details);
    }
} 