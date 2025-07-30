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
 * Pressure Sensor implementation for the Fog and Edge Computing System
 * 
 * This class simulates a pressure sensor that generates realistic atmospheric
 * pressure readings and transmits data via LoRaWAN to edge nodes.
 * Based on the research paper's IoT sensor implementation.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class PressureSensor extends BaseIoTDevice {
    
    private static final Logger logger = LoggerFactory.getLogger(PressureSensor.class);
    
    // Pressure sensor specific properties
    private double currentPressure;
    private double basePressure;
    private double pressureVariation;
    private Random random;
    
    // Sensor calibration
    private double calibrationOffset;
    private double accuracy;
    
    /**
     * Constructor for Pressure Sensor
     * 
     * @param deviceId Unique device identifier
     * @param networkManager Network manager for communication
     * @param metricsCollector Metrics collector for performance tracking
     */
    public PressureSensor(String deviceId, NetworkManager networkManager, MetricsCollector metricsCollector) {
        super(deviceId, "PRESSURE_SENSOR", networkManager, metricsCollector);
        
        this.random = new Random();
        this.basePressure = 1013.25 + (random.nextDouble() - 0.5) * 20.0; // 1003-1023 hPa base
        this.currentPressure = basePressure;
        this.pressureVariation = 5.0; // ±5 hPa variation
        this.calibrationOffset = (random.nextDouble() - 0.5) * 1.0; // ±0.5 hPa calibration
        this.accuracy = 0.1; // ±0.1 hPa accuracy
        
        logger.debug("Pressure sensor initialized: {} with base pressure: {} hPa", deviceId, basePressure);
    }
    
    @Override
    protected void initializeDevice() {
        logger.debug("Initializing pressure sensor: {}", deviceId);
        
        // Set device-specific configuration
        configuration.put("sensorType", "BMP280");
        configuration.put("measurementRange", "300.0 to 1100.0");
        configuration.put("resolution", "0.01 hPa");
        configuration.put("responseTime", "1.0"); // seconds
        configuration.put("samplingRate", "1.0"); // Hz
        
        // Initialize sensor calibration
        performCalibration();
        
        logger.debug("Pressure sensor {} initialized successfully", deviceId);
    }
    
    @Override
    protected void cleanupDevice() {
        logger.debug("Cleaning up pressure sensor: {}", deviceId);
        
        // Save calibration data
        saveCalibrationData();
        
        logger.debug("Pressure sensor {} cleanup completed", deviceId);
    }
    
    @Override
    public Object generateData() {
        try {
            // Generate realistic pressure reading with environmental variations
            double environmentalVariation = Math.sin(System.currentTimeMillis() / 20000.0) * pressureVariation;
            double randomNoise = (random.nextDouble() - 0.5) * 0.2; // ±0.1 hPa noise
            double weatherEffect = calculateWeatherEffect();
            
            // Calculate new pressure
            currentPressure = basePressure + environmentalVariation + randomNoise + weatherEffect + calibrationOffset;
            
            // Ensure pressure stays within valid range
            currentPressure = Math.max(300.0, Math.min(1100.0, currentPressure));
            
            // Add measurement noise based on sensor accuracy
            double measurementNoise = (random.nextDouble() - 0.5) * accuracy;
            double measuredPressure = currentPressure + measurementNoise;
            
            // Create pressure data object
            Map<String, Object> pressureData = new HashMap<>();
            pressureData.put("deviceId", deviceId);
            pressureData.put("deviceType", "PRESSURE_SENSOR");
            pressureData.put("timestamp", System.currentTimeMillis());
            pressureData.put("pressure", Math.round(measuredPressure * 100.0) / 100.0); // Round to 2 decimals
            pressureData.put("unit", "hPa");
            pressureData.put("accuracy", accuracy);
            pressureData.put("batteryLevel", batteryLevel);
            pressureData.put("signalStrength", signalStrength);
            pressureData.put("status", status);
            
            // Update total data generated
            totalDataGenerated.addAndGet(pressureData.toString().getBytes().length);
            
            logger.debug("Pressure sensor {} generated reading: {} hPa", deviceId, measuredPressure);
            
            return pressureData;
            
        } catch (Exception e) {
            logger.error("Error generating pressure data for device: {}", deviceId, e);
            return null;
        }
    }
    
    /**
     * Calculate weather effect on pressure
     * 
     * @return Pressure variation due to weather conditions
     */
    private double calculateWeatherEffect() {
        long currentTime = System.currentTimeMillis();
        long hoursSinceMidnight = (currentTime / (1000 * 60 * 60)) % 24;
        
        // Simulate weather-related pressure changes
        double weatherEffect = Math.sin((hoursSinceMidnight - 12) * Math.PI / 12) * 2.0; // ±2 hPa daily variation
        
        return weatherEffect;
    }
    
    /**
     * Perform sensor calibration
     */
    private void performCalibration() {
        logger.debug("Performing calibration for pressure sensor: {}", deviceId);
        
        // Simulate calibration process
        double[] calibrationReadings = new double[10];
        for (int i = 0; i < 10; i++) {
            calibrationReadings[i] = basePressure + (random.nextDouble() - 0.5) * 0.1;
        }
        
        // Calculate average and update calibration offset
        double average = 0.0;
        for (double reading : calibrationReadings) {
            average += reading;
        }
        average /= calibrationReadings.length;
        
        calibrationOffset = basePressure - average;
        
        logger.debug("Pressure sensor {} calibrated with offset: {} hPa", deviceId, calibrationOffset);
    }
    
    /**
     * Save calibration data
     */
    private void saveCalibrationData() {
        // In a real implementation, this would save to persistent storage
        logger.debug("Calibration data saved for pressure sensor: {}", deviceId);
    }
    
    /**
     * Get current pressure reading
     * 
     * @return Current pressure in hPa
     */
    public double getCurrentPressure() {
        return currentPressure;
    }
    
    /**
     * Get base pressure for this sensor
     * 
     * @return Base pressure in hPa
     */
    public double getBasePressure() {
        return basePressure;
    }
    
    /**
     * Get sensor accuracy
     * 
     * @return Sensor accuracy in hPa
     */
    public double getAccuracy() {
        return accuracy;
    }
    
    /**
     * Get calibration offset
     * 
     * @return Calibration offset in hPa
     */
    public double getCalibrationOffset() {
        return calibrationOffset;
    }
    
    @Override
    public Map<String, Object> getPerformanceMetrics() {
        Map<String, Object> metrics = super.getPerformanceMetrics();
        
        // Add pressure-specific metrics
        metrics.put("currentPressure", currentPressure);
        metrics.put("basePressure", basePressure);
        metrics.put("accuracy", accuracy);
        metrics.put("calibrationOffset", calibrationOffset);
        metrics.put("pressureVariation", pressureVariation);
        
        return metrics;
    }
    
    @Override
    public IoTDevice.DiagnosticResult performDiagnostic() {
        IoTDevice.DiagnosticResult baseResult = super.performDiagnostic();
        
        Map<String, Object> details = new HashMap<>(baseResult.getDetails());
        boolean passed = baseResult.isPassed();
        String message = baseResult.getMessage();
        
        // Add pressure-specific diagnostic checks
        if (currentPressure < 300.0 || currentPressure > 1100.0) {
            passed = false;
            message = "Pressure reading out of range";
        }
        details.put("currentPressure", currentPressure);
        details.put("pressureRange", "300.0 to 1100.0 hPa");
        
        if (Math.abs(calibrationOffset) > 2.0) {
            passed = false;
            message = "Calibration offset too large";
        }
        details.put("calibrationOffset", calibrationOffset);
        
        return new IoTDevice.DiagnosticResult(passed, message, details);
    }
} 