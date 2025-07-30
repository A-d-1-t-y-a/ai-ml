package com.nci.fogedge.iot;

import com.nci.fogedge.network.NetworkManager;
import com.nci.fogedge.utils.MetricsCollector;
import com.nci.fogedge.utils.PerformanceMetrics;

/**
 * IoT Device Interface for the Fog and Edge Computing System
 * 
 * This interface defines the contract for all IoT devices including sensors and actuators.
 * It provides methods for device lifecycle management, data collection, and communication.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public interface IoTDevice {
    
    /**
     * Get the unique device identifier
     * 
     * @return Device ID
     */
    String getDeviceId();
    
    /**
     * Get the device type
     * 
     * @return Device type
     */
    String getDeviceType();
    
    /**
     * Get the device location
     * 
     * @return Device location
     */
    String getLocation();
    
    /**
     * Start the device
     */
    void start();
    
    /**
     * Stop the device
     */
    void stop();
    
    /**
     * Check if the device is running
     * 
     * @return True if device is running
     */
    boolean isRunning();
    
    /**
     * Check if the device is healthy
     * 
     * @return True if device is healthy
     */
    boolean isHealthy();
    
    /**
     * Collect sensor data
     * 
     * @return Collected data as string
     */
    String collectData();
    
    /**
     * Transmit data to edge nodes
     * 
     * @param data Data to transmit
     * @return True if transmission successful
     */
    boolean transmitData(String data);
    
    /**
     * Get device status
     * 
     * @return Device status information
     */
    String getStatus();
    
    /**
     * Get device metrics
     * 
     * @return Device performance metrics
     */
    PerformanceMetrics getMetrics();
    
    /**
     * Update device configuration
     * 
     * @param config Configuration parameters
     */
    void updateConfiguration(java.util.Map<String, Object> config);
    
    /**
     * Get battery level (for battery-powered devices)
     * 
     * @return Battery level percentage (0-100)
     */
    double getBatteryLevel();
    
    /**
     * Get signal strength
     * 
     * @return Signal strength in dBm
     */
    double getSignalStrength();
    
    /**
     * Get data transmission rate
     * 
     * @return Transmission rate in bytes per second
     */
    double getTransmissionRate();
    
    /**
     * Get error count
     * 
     * @return Number of errors encountered
     */
    int getErrorCount();
    
    /**
     * Reset device error count
     */
    void resetErrorCount();
    
    /**
     * Get last data collection timestamp
     * 
     * @return Timestamp of last data collection
     */
    long getLastDataCollectionTime();
    
    /**
     * Get last transmission timestamp
     * 
     * @return Timestamp of last transmission
     */
    long getLastTransmissionTime();
} 