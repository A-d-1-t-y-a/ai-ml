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
 * Light Sensor implementation for the Fog and Edge Computing System
 * 
 * This class simulates a light sensor that generates realistic light intensity
 * readings and transmits data via LoRaWAN to edge nodes.
 * Based on the research paper's IoT sensor implementation.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class LightSensor extends BaseIoTDevice {
    
    private static final Logger logger = LoggerFactory.getLogger(LightSensor.class);
    
    // Light sensor specific properties
    private double currentLightLevel;
    private double baseLightLevel;
    private double lightVariation;
    private Random random;
    
    // Sensor calibration
    private double calibrationOffset;
    private double accuracy;
    
    /**
     * Constructor for Light Sensor
     * 
     * @param deviceId Unique device identifier
     * @param networkManager Network manager for communication
     * @param metricsCollector Metrics collector for performance tracking
     */
    public LightSensor(String deviceId, NetworkManager networkManager, MetricsCollector metricsCollector) {
        super(deviceId, "LIGHT_SENSOR", networkManager, metricsCollector);
        
        this.random = new Random();
        this.baseLightLevel = 500.0 + (random.nextDouble() - 0.5) * 200.0; // 400-600 lux base
        this.currentLightLevel = baseLightLevel;
        this.lightVariation = 100.0; // ±100 lux variation
        this.calibrationOffset = (random.nextDouble() - 0.5) * 10.0; // ±5 lux calibration
        this.accuracy = 1.0; // ±1 lux accuracy
        
        logger.debug("Light sensor initialized: {} with base light level: {} lux", deviceId, baseLightLevel);
    }
    
    @Override
    protected void initializeDevice() {
        logger.debug("Initializing light sensor: {}", deviceId);
        
        // Set device-specific configuration
        configuration.put("sensorType", "BH1750");
        configuration.put("measurementRange", "1.0 to 65535.0");
        configuration.put("resolution", "1.0 lux");
        configuration.put("responseTime", "0.5"); // seconds
        configuration.put("samplingRate", "2.0"); // Hz
        
        // Initialize sensor calibration
        performCalibration();
        
        logger.debug("Light sensor {} initialized successfully", deviceId);
    }
    
    @Override
    protected void cleanupDevice() {
        logger.debug("Cleaning up light sensor: {}", deviceId);
        
        // Save calibration data
        saveCalibrationData();
        
        logger.debug("Light sensor {} cleanup completed", deviceId);
    }
    
    @Override
    public String collectData() {
        try {
            // Generate realistic light reading with environmental variations
            double environmentalVariation = Math.sin(System.currentTimeMillis() / 8000.0) * lightVariation;
            double randomNoise = (random.nextDouble() - 0.5) * 5.0; // ±2.5 lux noise
            double timeOfDayEffect = calculateTimeOfDayEffect();
            
            // Calculate new light level
            currentLightLevel = baseLightLevel + environmentalVariation + randomNoise + timeOfDayEffect + calibrationOffset;
            
            // Ensure light level stays within valid range
            currentLightLevel = Math.max(1.0, Math.min(65535.0, currentLightLevel));
            
            // Add measurement noise based on sensor accuracy
            double measurementNoise = (random.nextDouble() - 0.5) * accuracy;
            double measuredLightLevel = currentLightLevel + measurementNoise;
            measuredLightLevel = Math.max(1.0, Math.min(65535.0, measuredLightLevel));
            
            // Create light data JSON string
            String lightData = String.format(
                "{\"deviceId\":\"%s\",\"deviceType\":\"LIGHT_SENSOR\",\"timestamp\":%d,\"lightLevel\":%.0f,\"unit\":\"lux\",\"accuracy\":%.1f,\"batteryLevel\":%.1f,\"signalStrength\":%.1f,\"status\":\"%s\"}",
                deviceId,
                System.currentTimeMillis(),
                Math.round(measuredLightLevel),
                accuracy,
                batteryLevel,
                signalStrength,
                status
            );
            
            // Update total data generated
            totalDataGenerated.addAndGet(lightData.getBytes().length);
            
            logger.debug("Light sensor {} generated reading: {} lux", deviceId, measuredLightLevel);
            
            return lightData;
            
        } catch (Exception e) {
            logger.error("Error generating light data for device: {}", deviceId, e);
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
                    logger.debug("Light sensor {} transmitted data successfully", deviceId);
                } else {
                    failedTransmissions.incrementAndGet();
                    logger.warn("Light sensor {} failed to transmit data", deviceId);
                }
                return success;
            }
            return false;
        } catch (Exception e) {
            failedTransmissions.incrementAndGet();
            logger.error("Error transmitting data from light sensor: {}", deviceId, e);
            return false;
        }
    }
    
    @Override
    public PerformanceMetrics getMetrics() {
        PerformanceMetrics metrics = new PerformanceMetrics(deviceId, "LIGHT_SENSOR");
        
        metrics.addMetric("batteryLevel", batteryLevel);
        metrics.addMetric("signalStrength", signalStrength);
        metrics.addMetric("transmissionRate", getTransmissionRate());
        metrics.addMetric("errorCount", getErrorCount());
        metrics.addMetric("currentLightLevel", currentLightLevel);
        metrics.addMetric("baseLightLevel", baseLightLevel);
        metrics.addMetric("accuracy", accuracy);
        metrics.addMetric("calibrationOffset", calibrationOffset);
        metrics.addMetric("lightVariation", lightVariation);
        metrics.addMetric("isHealthy", isHealthy());
        metrics.addMetric("isRunning", isRunning());
        
        return metrics;
    }
    
    @Override
    public void updateConfiguration(Map<String, Object> config) {
        if (config != null) {
            configuration.putAll(config);
            logger.debug("Light sensor {} configuration updated", deviceId);
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
     * Calculate time of day effect on light level
     * 
     * @return Light level variation due to time of day
     */
    private double calculateTimeOfDayEffect() {
        long currentTime = System.currentTimeMillis();
        long hoursSinceMidnight = (currentTime / (1000 * 60 * 60)) % 24;
        
        // Simulate daily light cycle (dark at night, bright during day)
        double timeEffect = Math.sin((hoursSinceMidnight - 6) * Math.PI / 12) * 1000.0; // ±1000 lux daily variation
        
        // Ensure minimum light level at night
        if (timeEffect < -500.0) {
            timeEffect = -500.0;
        }
        
        return timeEffect;
    }
    
    /**
     * Perform sensor calibration
     */
    private void performCalibration() {
        logger.debug("Performing calibration for light sensor: {}", deviceId);
        
        // Simulate calibration process
        double[] calibrationReadings = new double[10];
        for (int i = 0; i < 10; i++) {
            calibrationReadings[i] = baseLightLevel + (random.nextDouble() - 0.5) * 1.0;
        }
        
        // Calculate average and update calibration offset
        double average = 0.0;
        for (double reading : calibrationReadings) {
            average += reading;
        }
        average /= calibrationReadings.length;
        
        calibrationOffset = baseLightLevel - average;
        
        logger.debug("Light sensor {} calibrated with offset: {} lux", deviceId, calibrationOffset);
    }
    
    /**
     * Save calibration data
     */
    private void saveCalibrationData() {
        // In a real implementation, this would save to persistent storage
        logger.debug("Calibration data saved for light sensor: {}", deviceId);
    }
    
    /**
     * Get current light level reading
     * 
     * @return Current light level in lux
     */
    public double getCurrentLightLevel() {
        return currentLightLevel;
    }
    
    /**
     * Get base light level for this sensor
     * 
     * @return Base light level in lux
     */
    public double getBaseLightLevel() {
        return baseLightLevel;
    }
    
    /**
     * Get sensor accuracy
     * 
     * @return Sensor accuracy in lux
     */
    public double getAccuracy() {
        return accuracy;
    }
    
    /**
     * Get calibration offset
     * 
     * @return Calibration offset in lux
     */
    public double getCalibrationOffset() {
        return calibrationOffset;
    }
    
    @Override
    public DiagnosticResult performDiagnostic() {
        Map<String, Object> details = new HashMap<>();
        boolean passed = true;
        String message = "Light sensor diagnostic passed";
        
        // Add light-specific diagnostic checks
        if (currentLightLevel < 1.0 || currentLightLevel > 65535.0) {
            passed = false;
            message = "Light level reading out of range";
        }
        details.put("currentLightLevel", currentLightLevel);
        details.put("lightLevelRange", "1.0 to 65535.0 lux");
        
        if (Math.abs(calibrationOffset) > 20.0) {
            passed = false;
            message = "Calibration offset too large";
        }
        details.put("calibrationOffset", calibrationOffset);
        
        details.put("batteryLevel", batteryLevel);
        details.put("signalStrength", signalStrength);
        details.put("status", status);
        
        return passed ? DiagnosticResult.success(message, details) : DiagnosticResult.failure(message, details);
    }
} 