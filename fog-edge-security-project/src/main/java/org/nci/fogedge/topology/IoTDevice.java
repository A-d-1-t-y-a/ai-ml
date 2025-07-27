package org.nci.fogedge.topology;

import org.cloudbus.cloudsim.Log;
import org.nci.fogedge.security.SecurityManager;
import org.nci.fogedge.security.SecurityLevel;

import java.util.Random;

/**
 * Represents an IoT device in the fog computing architecture.
 * IoT devices are the lowest level in the hierarchy and generate data
 * that is processed by edge and fog nodes.
 */
public class IoTDevice {
    private String deviceId;
    private EdgeNode parentEdge;
    private String wirelessTechnology;
    private SecurityManager securityManager;
    private double energyConsumption;
    private double securityOverhead;
    private double dataGenerationRate; // in KB/s
    private SecurityLevel securityLevel;
    
    // Random for simulation purposes
    private Random random = new Random();
    
    /**
     * Creates a new IoT device with specified parameters
     * 
     * @param deviceId Unique identifier for the device
     * @param parentEdge The edge node this device connects to
     * @param wirelessTechnology The wireless technology used (WiFi, BLE, LoRaWAN)
     * @param securityManager The security manager for encryption/authentication
     */
    public IoTDevice(String deviceId, EdgeNode parentEdge, String wirelessTechnology, SecurityManager securityManager) {
        this.deviceId = deviceId;
        this.parentEdge = parentEdge;
        this.wirelessTechnology = wirelessTechnology;
        this.securityManager = securityManager;
        
        // Initialize metrics
        this.energyConsumption = 0.0;
        this.securityOverhead = 0.0;
        
        // Set data generation rate based on wireless technology
        switch (wirelessTechnology) {
            case "WiFi":
                this.dataGenerationRate = 50.0 + random.nextDouble() * 50.0; // 50-100 KB/s
                this.securityLevel = SecurityLevel.HIGH;
                break;
            case "BLE":
                this.dataGenerationRate = 10.0 + random.nextDouble() * 15.0; // 10-25 KB/s
                this.securityLevel = SecurityLevel.MEDIUM;
                break;
            case "LoRaWAN":
                this.dataGenerationRate = 0.5 + random.nextDouble() * 1.5; // 0.5-2 KB/s
                this.securityLevel = SecurityLevel.LOW;
                break;
            default:
                this.dataGenerationRate = 10.0;
                this.securityLevel = SecurityLevel.MEDIUM;
        }
        
        // Register with parent edge node
        parentEdge.registerIoTDevice(this);
        
        Log.printLine("IoT Device " + deviceId + " created with " + wirelessTechnology + 
                " technology and data generation rate of " + String.format("%.2f", dataGenerationRate) + " KB/s");
    }
    
    /**
     * Generates data and sends it to the parent edge node
     * 
     * @param dataSize Size of data in KB
     * @return Processing time in ms
     */
    public double generateAndSendData(double dataSize) {
        double startTime = System.currentTimeMillis();
        
        // Apply security measures based on security level
        byte[] data = new byte[(int) (dataSize * 1024)]; // Convert KB to bytes
        random.nextBytes(data); // Generate random data
        
        // Apply encryption if security is enabled
        if (securityManager.isSecurityEnabled()) {
            double beforeEncryption = System.currentTimeMillis();
            byte[] encryptedData = securityManager.encryptData(data, securityLevel);
            double encryptionTime = System.currentTimeMillis() - beforeEncryption;
            
            // Update security overhead
            this.securityOverhead += encryptionTime;
            
            // Calculate energy consumption for encryption
            // Energy model: E = k * data_size * security_level_factor
            double encryptionEnergy = 0.001 * dataSize * securityLevel.getFactor();
            this.energyConsumption += encryptionEnergy;
            
            // Send data to edge node
            parentEdge.receiveData(deviceId, encryptedData, true);
        } else {
            // Send unencrypted data
            parentEdge.receiveData(deviceId, data, false);
        }
        
        // Calculate transmission energy based on wireless technology
        double transmissionEnergy = calculateTransmissionEnergy(dataSize);
        this.energyConsumption += transmissionEnergy;
        
        return System.currentTimeMillis() - startTime;
    }
    
    /**
     * Calculates energy consumption for data transmission based on wireless technology
     * 
     * @param dataSize Size of data in KB
     * @return Energy consumption in mJ
     */
    private double calculateTransmissionEnergy(double dataSize) {
        // Energy models based on wireless technology
        switch (wirelessTechnology) {
            case "WiFi":
                return 0.05 * dataSize; // 0.05 mJ/KB
            case "BLE":
                return 0.02 * dataSize; // 0.02 mJ/KB
            case "LoRaWAN":
                return 0.01 * dataSize; // 0.01 mJ/KB
            default:
                return 0.03 * dataSize; // Default
        }
    }
    
    /**
     * Simulates device operation for a specified duration
     * 
     * @param durationMs Simulation duration in milliseconds
     */
    public void simulateOperation(double durationMs) {
        // Calculate how much data would be generated during this time
        double dataGenerated = dataGenerationRate * (durationMs / 1000.0);
        
        // Send data in chunks
        double chunkSize = 10.0; // 10 KB chunks
        int numChunks = (int) Math.ceil(dataGenerated / chunkSize);
        
        for (int i = 0; i < numChunks; i++) {
            double actualChunkSize = Math.min(chunkSize, dataGenerated - (i * chunkSize));
            if (actualChunkSize > 0) {
                generateAndSendData(actualChunkSize);
            }
        }
    }
    
    // Getters
    public String getDeviceId() {
        return deviceId;
    }
    
    public String getWirelessTechnology() {
        return wirelessTechnology;
    }
    
    public double getEnergyConsumption() {
        return energyConsumption;
    }
    
    public double getSecurityOverhead() {
        return securityOverhead;
    }
    
    public double getDataGenerationRate() {
        return dataGenerationRate;
    }
    
    public SecurityLevel getSecurityLevel() {
        return securityLevel;
    }
}
