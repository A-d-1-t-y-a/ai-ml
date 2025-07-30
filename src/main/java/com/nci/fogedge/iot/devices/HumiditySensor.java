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
    public Object generateData() {
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
            
            // Create humidity data object
            Map<String, Object> humidityData = new HashMap<>();
            humidityData.put("deviceId", deviceId);
            humidityData.put("deviceType", "HUMIDITY_SENSOR");
            humidityData.put("timestamp", System.currentTimeMillis());
            humidityData.put("humidity", Math.round(measuredHumidity * 10.0) / 10.0); // Round to 1 decimal
            humidityData.put("unit", "percent");
            humidityData.put("accuracy", accuracy);
            humidityData.put("batteryLevel", batteryLevel);
            humidityData.put("signalStrength", signalStrength);
            humidityData.put("status", status);
            
            // Update total data generated
            totalDataGenerated.addAndGet(humidityData.toString().getBytes().length);
            
            logger.debug("Humidity sensor {} generated reading: {}%", deviceId, measuredHumidity);
            
            return humidityData;
            
        } catch (Exception e) {
            logger.error("Error generating humidity data for device: {}", deviceId, e);
            return null;
        }
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
    public Map<String, Object> getPerformanceMetrics() {
        Map<String, Object> metrics = super.getPerformanceMetrics();
        
        // Add humidity-specific metrics
        metrics.put("currentHumidity", currentHumidity);
        metrics.put("baseHumidity", baseHumidity);
        metrics.put("accuracy", accuracy);
        metrics.put("calibrationOffset", calibrationOffset);
        metrics.put("humidityVariation", humidityVariation);
        
        return metrics;
    }
    
    @Override
    public IoTDevice.DiagnosticResult performDiagnostic() {
        IoTDevice.DiagnosticResult baseResult = super.performDiagnostic();
        
        Map<String, Object> details = new HashMap<>(baseResult.getDetails());
        boolean passed = baseResult.isPassed();
        String message = baseResult.getMessage();
        
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
        
        return new IoTDevice.DiagnosticResult(passed, message, details);
    }
} 