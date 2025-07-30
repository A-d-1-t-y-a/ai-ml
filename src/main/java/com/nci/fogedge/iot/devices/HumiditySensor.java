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
 * Humidity Sensor implementation for the Fog and Edge Computing System
 * 
 * This class simulates a humidity sensor that generates realistic humidity
 * readings with environmental variations and transmits data via LoRaWAN to edge nodes.
 * Based on the research paper's IoT sensor implementation.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class HumiditySensor extends BaseIoTDevice {
    
    private static final Logger logger = LoggerFactory.getLogger(HumiditySensor.class);
    
    // Humidity sensor specific properties
    private double currentHumidity;
    private double baseHumidity;
    private double humidityVariation;
    private Random random;
    
    // Sensor calibration
    private double calibrationOffset;
    private double accuracy;
    
    /**
     * Constructor for Humidity Sensor
     * 
     * @param deviceId Unique device identifier
     * @param networkManager Network manager for communication
     * @param metricsCollector Metrics collector for performance tracking
     */
    public HumiditySensor(String deviceId, NetworkManager networkManager, MetricsCollector metricsCollector) {
        super(deviceId, "HUMIDITY_SENSOR", networkManager, metricsCollector);
        
        this.random = new Random();
        this.baseHumidity = 50.0 + (random.nextDouble() - 0.5) * 20.0; // 40-60% base
        this.currentHumidity = baseHumidity;
        this.humidityVariation = 5.0; // ±5% variation
        this.calibrationOffset = (random.nextDouble() - 0.5) * 2.0; // ±1% calibration
        this.accuracy = 0.5; // ±0.5% accuracy
        
        logger.debug("Humidity sensor initialized: {} with base humidity: {}%", deviceId, baseHumidity);
    }
    
    @Override
    protected void initializeDevice() {
        logger.debug("Initializing humidity sensor: {}", deviceId);
        
        // Set device-specific configuration
        configuration.put("sensorType", "DHT22");
        configuration.put("measurementRange", "0.0 to 100.0");
        configuration.put("resolution", "0.1%");
        configuration.put("responseTime", "2.0"); // seconds
        configuration.put("samplingRate", "1.0"); // Hz
        
        // Initialize sensor calibration
        performCalibration();
        
        logger.debug("Humidity sensor {} initialized successfully", deviceId);
    }
    
    @Override
    protected void cleanupDevice() {
        logger.debug("Cleaning up humidity sensor: {}", deviceId);
        
        // Save calibration data
        saveCalibrationData();
        
        logger.debug("Humidity sensor {} cleanup completed", deviceId);
    }
    
    @Override
    public String collectData() {
        try {
            // Generate realistic humidity reading with environmental variations
            double environmentalVariation = Math.sin(System.currentTimeMillis() / 15000.0) * humidityVariation;
            double randomNoise = (random.nextDouble() - 0.5) * 1.0; // ±0.5% noise
            double timeOfDayEffect = calculateTimeOfDayEffect();
            
            // Calculate new humidity
            currentHumidity = baseHumidity + environmentalVariation + randomNoise + timeOfDayEffect + calibrationOffset;
            
            // Ensure humidity stays within valid range (0-100%)
            currentHumidity = Math.max(0.0, Math.min(100.0, currentHumidity));
            
            // Add measurement noise based on sensor accuracy
            double measurementNoise = (random.nextDouble() - 0.5) * accuracy;
            double measuredHumidity = currentHumidity + measurementNoise;
            measuredHumidity = Math.max(0.0, Math.min(100.0, measuredHumidity));
            
            // Create humidity data JSON string
            String humidityData = String.format(
                "{\"deviceId\":\"%s\",\"deviceType\":\"HUMIDITY_SENSOR\",\"timestamp\":%d,\"humidity\":%.1f,\"unit\":\"percent\",\"accuracy\":%.1f,\"batteryLevel\":%.1f,\"signalStrength\":%.1f,\"status\":\"%s\"}",
                deviceId,
                System.currentTimeMillis(),
                Math.round(measuredHumidity * 10.0) / 10.0,
                accuracy,
                batteryLevel,
                signalStrength,
                status
            );
            
            // Update total data generated
            totalDataGenerated.addAndGet(humidityData.getBytes().length);
            
            logger.debug("Humidity sensor {} generated reading: {}%", deviceId, measuredHumidity);
            
            return humidityData;
            
        } catch (Exception e) {
            logger.error("Error generating humidity data for device: {}", deviceId, e);
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
                    logger.debug("Humidity sensor {} transmitted data successfully", deviceId);
                } else {
                    failedTransmissions.incrementAndGet();
                    logger.warn("Humidity sensor {} failed to transmit data", deviceId);
                }
                return success;
            }
            return false;
        } catch (Exception e) {
            failedTransmissions.incrementAndGet();
            logger.error("Error transmitting data from humidity sensor: {}", deviceId, e);
            return false;
        }
    }
    
    @Override
    public PerformanceMetrics getMetrics() {
        PerformanceMetrics metrics = new PerformanceMetrics(deviceId, "HUMIDITY_SENSOR");
        
        metrics.addMetric("batteryLevel", batteryLevel);
        metrics.addMetric("signalStrength", signalStrength);
        metrics.addMetric("transmissionRate", getTransmissionRate());
        metrics.addMetric("errorCount", getErrorCount());
        metrics.addMetric("currentHumidity", currentHumidity);
        metrics.addMetric("baseHumidity", baseHumidity);
        metrics.addMetric("accuracy", accuracy);
        metrics.addMetric("calibrationOffset", calibrationOffset);
        metrics.addMetric("humidityVariation", humidityVariation);
        metrics.addMetric("isHealthy", isHealthy());
        metrics.addMetric("isRunning", isRunning());
        
        return metrics;
    }
    
    @Override
    public void updateConfiguration(Map<String, Object> config) {
        if (config != null) {
            configuration.putAll(config);
            logger.debug("Humidity sensor {} configuration updated", deviceId);
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
     * Calculate time of day effect on humidity
     * 
     * @return Humidity variation due to time of day
     */
    private double calculateTimeOfDayEffect() {
        long currentTime = System.currentTimeMillis();
        long hoursSinceMidnight = (currentTime / (1000 * 60 * 60)) % 24;
        
        // Simulate daily humidity cycle (higher at night, lower during day)
        double timeEffect = Math.sin((hoursSinceMidnight - 6) * Math.PI / 12) * 3.0; // ±3% daily variation
        
        return timeEffect;
    }
    
    /**
     * Perform sensor calibration
     */
    private void performCalibration() {
        logger.debug("Performing calibration for humidity sensor: {}", deviceId);
        
        // Simulate calibration process
        double[] calibrationReadings = new double[10];
        for (int i = 0; i < 10; i++) {
            calibrationReadings[i] = baseHumidity + (random.nextDouble() - 0.5) * 0.5;
        }
        
        // Calculate average and update calibration offset
        double average = 0.0;
        for (double reading : calibrationReadings) {
            average += reading;
        }
        average /= calibrationReadings.length;
        
        calibrationOffset = baseHumidity - average;
        
        logger.debug("Humidity sensor {} calibrated with offset: {}%", deviceId, calibrationOffset);
    }
    
    /**
     * Save calibration data
     */
    private void saveCalibrationData() {
        // In a real implementation, this would save to persistent storage
        logger.debug("Calibration data saved for humidity sensor: {}", deviceId);
    }
    
    /**
     * Get current humidity reading
     * 
     * @return Current humidity percentage
     */
    public double getCurrentHumidity() {
        return currentHumidity;
    }
    
    /**
     * Get base humidity for this sensor
     * 
     * @return Base humidity percentage
     */
    public double getBaseHumidity() {
        return baseHumidity;
    }
    
    /**
     * Get sensor accuracy
     * 
     * @return Sensor accuracy percentage
     */
    public double getAccuracy() {
        return accuracy;
    }
    
    /**
     * Get calibration offset
     * 
     * @return Calibration offset percentage
     */
    public double getCalibrationOffset() {
        return calibrationOffset;
    }
    
    @Override
    public DiagnosticResult performDiagnostic() {
        Map<String, Object> details = new HashMap<>();
        boolean passed = true;
        String message = "Humidity sensor diagnostic passed";
        
        // Add humidity-specific diagnostic checks
        if (currentHumidity < 0.0 || currentHumidity > 100.0) {
            passed = false;
            message = "Humidity reading out of range";
        }
        details.put("currentHumidity", currentHumidity);
        details.put("humidityRange", "0.0 to 100.0%");
        
        if (Math.abs(calibrationOffset) > 5.0) {
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