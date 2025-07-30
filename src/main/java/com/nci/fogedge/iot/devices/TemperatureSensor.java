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
 * Temperature Sensor implementation for the Fog and Edge Computing System
 * 
 * This class simulates a temperature sensor that generates realistic temperature
 * readings with environmental variations and transmits data via LoRaWAN to edge nodes.
 * Based on the research paper's IoT sensor implementation.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class TemperatureSensor extends BaseIoTDevice {
    
    private static final Logger logger = LoggerFactory.getLogger(TemperatureSensor.class);
    
    // Temperature sensor specific properties
    private double currentTemperature;
    private double baseTemperature;
    private double temperatureVariation;
    private Random random;
    
    // Sensor calibration
    private double calibrationOffset;
    private double accuracy;
    
    /**
     * Constructor for Temperature Sensor
     * 
     * @param deviceId Unique device identifier
     * @param networkManager Network manager for communication
     * @param metricsCollector Metrics collector for performance tracking
     */
    public TemperatureSensor(String deviceId, NetworkManager networkManager, MetricsCollector metricsCollector) {
        super(deviceId, "TEMPERATURE_SENSOR", networkManager, metricsCollector);
        
        this.random = new Random();
        this.baseTemperature = 20.0 + (random.nextDouble() - 0.5) * 10.0; // 15-25°C base
        this.currentTemperature = baseTemperature;
        this.temperatureVariation = 2.0; // ±2°C variation
        this.calibrationOffset = (random.nextDouble() - 0.5) * 0.5; // ±0.25°C calibration
        this.accuracy = 0.1; // ±0.1°C accuracy
        
        logger.debug("Temperature sensor initialized: {} with base temperature: {}°C", deviceId, baseTemperature);
    }
    
    @Override
    protected void initializeDevice() {
        logger.debug("Initializing temperature sensor: {}", deviceId);
        
        // Set device-specific configuration
        configuration.put("sensorType", "DHT22");
        configuration.put("measurementRange", "-40.0 to 80.0");
        configuration.put("resolution", "0.1°C");
        configuration.put("responseTime", "2.0"); // seconds
        configuration.put("samplingRate", "1.0"); // Hz
        
        // Initialize sensor calibration
        performCalibration();
        
        logger.debug("Temperature sensor {} initialized successfully", deviceId);
    }
    
    @Override
    protected void cleanupDevice() {
        logger.debug("Cleaning up temperature sensor: {}", deviceId);
        
        // Save calibration data
        saveCalibrationData();
        
        logger.debug("Temperature sensor {} cleanup completed", deviceId);
    }
    
    @Override
    public String collectData() {
        try {
            // Generate realistic temperature reading with environmental variations
            double environmentalVariation = Math.sin(System.currentTimeMillis() / 10000.0) * temperatureVariation;
            double randomNoise = (random.nextDouble() - 0.5) * 0.2; // ±0.1°C noise
            double timeOfDayEffect = calculateTimeOfDayEffect();
            
            // Calculate new temperature
            currentTemperature = baseTemperature + environmentalVariation + randomNoise + timeOfDayEffect + calibrationOffset;
            
            // Add measurement noise based on sensor accuracy
            double measurementNoise = (random.nextDouble() - 0.5) * accuracy;
            double measuredTemperature = currentTemperature + measurementNoise;
            
            // Create temperature data JSON string
            String temperatureData = String.format(
                "{\"deviceId\":\"%s\",\"deviceType\":\"TEMPERATURE_SENSOR\",\"timestamp\":%d,\"temperature\":%.1f,\"unit\":\"celsius\",\"accuracy\":%.1f,\"batteryLevel\":%.1f,\"signalStrength\":%.1f,\"status\":\"%s\"}",
                deviceId,
                System.currentTimeMillis(),
                Math.round(measuredTemperature * 10.0) / 10.0,
                accuracy,
                batteryLevel,
                signalStrength,
                status
            );
            
            // Update total data generated
            totalDataGenerated.addAndGet(temperatureData.getBytes().length);
            
            logger.debug("Temperature sensor {} generated reading: {}°C", deviceId, measuredTemperature);
            
            return temperatureData;
            
        } catch (Exception e) {
            logger.error("Error generating temperature data for device: {}", deviceId, e);
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
                    logger.debug("Temperature sensor {} transmitted data successfully", deviceId);
                } else {
                    failedTransmissions.incrementAndGet();
                    logger.warn("Temperature sensor {} failed to transmit data", deviceId);
                }
                return success;
            }
            return false;
        } catch (Exception e) {
            failedTransmissions.incrementAndGet();
            logger.error("Error transmitting data from temperature sensor: {}", deviceId, e);
            return false;
        }
    }
    
    @Override
    public PerformanceMetrics getMetrics() {
        PerformanceMetrics metrics = new PerformanceMetrics(deviceId, "TEMPERATURE_SENSOR");
        
        metrics.addMetric("batteryLevel", batteryLevel);
        metrics.addMetric("signalStrength", signalStrength);
        metrics.addMetric("transmissionRate", getTransmissionRate());
        metrics.addMetric("errorCount", getErrorCount());
        metrics.addMetric("currentTemperature", currentTemperature);
        metrics.addMetric("baseTemperature", baseTemperature);
        metrics.addMetric("accuracy", accuracy);
        metrics.addMetric("calibrationOffset", calibrationOffset);
        metrics.addMetric("temperatureVariation", temperatureVariation);
        metrics.addMetric("isHealthy", isHealthy());
        metrics.addMetric("isRunning", isRunning());
        
        return metrics;
    }
    
    @Override
    public void updateConfiguration(Map<String, Object> config) {
        if (config != null) {
            configuration.putAll(config);
            logger.debug("Temperature sensor {} configuration updated", deviceId);
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
     * Calculate time of day effect on temperature
     * 
     * @return Temperature variation due to time of day
     */
    private double calculateTimeOfDayEffect() {
        long currentTime = System.currentTimeMillis();
        long hoursSinceMidnight = (currentTime / (1000 * 60 * 60)) % 24;
        
        // Simulate daily temperature cycle (colder at night, warmer during day)
        double timeEffect = Math.sin((hoursSinceMidnight - 6) * Math.PI / 12) * 3.0; // ±3°C daily variation
        
        return timeEffect;
    }
    
    /**
     * Perform sensor calibration
     */
    private void performCalibration() {
        logger.debug("Performing calibration for temperature sensor: {}", deviceId);
        
        // Simulate calibration process
        double[] calibrationReadings = new double[10];
        for (int i = 0; i < 10; i++) {
            calibrationReadings[i] = baseTemperature + (random.nextDouble() - 0.5) * 0.1;
        }
        
        // Calculate average and update calibration offset
        double average = 0.0;
        for (double reading : calibrationReadings) {
            average += reading;
        }
        average /= calibrationReadings.length;
        
        calibrationOffset = baseTemperature - average;
        
        logger.debug("Temperature sensor {} calibrated with offset: {}°C", deviceId, calibrationOffset);
    }
    
    /**
     * Save calibration data
     */
    private void saveCalibrationData() {
        // In a real implementation, this would save to persistent storage
        logger.debug("Calibration data saved for temperature sensor: {}", deviceId);
    }
    
    /**
     * Get current temperature reading
     * 
     * @return Current temperature in Celsius
     */
    public double getCurrentTemperature() {
        return currentTemperature;
    }
    
    /**
     * Get base temperature for this sensor
     * 
     * @return Base temperature in Celsius
     */
    public double getBaseTemperature() {
        return baseTemperature;
    }
    
    /**
     * Get sensor accuracy
     * 
     * @return Sensor accuracy in Celsius
     */
    public double getAccuracy() {
        return accuracy;
    }
    
    /**
     * Get calibration offset
     * 
     * @return Calibration offset in Celsius
     */
    public double getCalibrationOffset() {
        return calibrationOffset;
    }
    
    @Override
    public DiagnosticResult performDiagnostic() {
        Map<String, Object> details = new HashMap<>();
        boolean passed = true;
        String message = "Temperature sensor diagnostic passed";
        
        // Add temperature-specific diagnostic checks
        if (currentTemperature < -40.0 || currentTemperature > 80.0) {
            passed = false;
            message = "Temperature reading out of range";
        }
        details.put("currentTemperature", currentTemperature);
        details.put("temperatureRange", "-40.0 to 80.0°C");
        
        if (Math.abs(calibrationOffset) > 1.0) {
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