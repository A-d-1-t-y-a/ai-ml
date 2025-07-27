package org.nci.fogedge.topology;

import java.util.logging.Logger;
import org.nci.fogedge.security.SecurityManager;
import org.nci.fogedge.security.SecurityLevel;
import org.nci.fogedge.util.DataProcessor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Represents an Edge Node in the fog computing architecture.
 * Edge nodes are intermediate nodes that process data from IoT devices
 * before forwarding to fog nodes.
 */
public class EdgeNode {
    private String nodeId;
    private FogNode parentFog;
    private SecurityManager securityManager;
    private List<IoTDevice> connectedDevices;
    private Map<String, byte[]> dataCache;
    private double processingTime;
    private double energyConsumption;
    private double securityOverhead;
    private double processingCapacity; // MIPS
    private double storageCapacity;    // MB
    private double availableBandwidth; // Mbps
    
    /**
     * Creates a new Edge Node with specified parameters
     * 
     * @param nodeId Unique identifier for the edge node
     * @param parentFog The fog node this edge node connects to
     * @param securityManager The security manager for encryption/authentication
     */
    public EdgeNode(String nodeId, FogNode parentFog, SecurityManager securityManager) {
        this.nodeId = nodeId;
        this.parentFog = parentFog;
        this.securityManager = securityManager;
        this.connectedDevices = new ArrayList<>();
        this.dataCache = new HashMap<>();
        
        // Initialize metrics
        this.processingTime = 0.0;
        this.energyConsumption = 0.0;
        this.securityOverhead = 0.0;
        
        // Set edge node capabilities
        this.processingCapacity = 2000.0; // 2000 MIPS
        this.storageCapacity = 1000.0;    // 1000 MB
        this.availableBandwidth = 100.0;  // 100 Mbps
        
        // Register with parent fog node
        parentFog.registerEdgeNode(this);
        
        Log.printLine("Edge Node " + nodeId + " created with processing capacity of " + 
                processingCapacity + " MIPS and storage capacity of " + storageCapacity + " MB");
    }
    
    /**
     * Registers an IoT device with this edge node
     * 
     * @param device The IoT device to register
     */
    public void registerIoTDevice(IoTDevice device) {
        connectedDevices.add(device);
        Log.printLine("IoT Device " + device.getDeviceId() + " registered with Edge Node " + nodeId);
    }
    
    /**
     * Receives data from an IoT device, processes it, and forwards to fog node if necessary
     * 
     * @param deviceId ID of the sending device
     * @param data The data received
     * @param isEncrypted Whether the data is encrypted
     */
    public void receiveData(String deviceId, byte[] data, boolean isEncrypted) {
        double startTime = System.currentTimeMillis();
        
        // Decrypt data if encrypted
        byte[] processedData = data;
        if (isEncrypted && securityManager.isSecurityEnabled()) {
            double beforeDecryption = System.currentTimeMillis();
            processedData = securityManager.decryptData(data);
            double decryptionTime = System.currentTimeMillis() - beforeDecryption;
            
            // Update security overhead
            this.securityOverhead += decryptionTime;
        }
        
        // Process data at the edge
        double beforeProcessing = System.currentTimeMillis();
        processedData = DataProcessor.processDataAtEdge(processedData);
        double processingTime = System.currentTimeMillis() - beforeProcessing;
        
        // Update processing time
        this.processingTime += processingTime;
        
        // Calculate energy consumption for processing
        // Energy model: E = k * data_size * processing_time
        double processingEnergy = 0.002 * (data.length / 1024.0) * (processingTime / 1000.0);
        this.energyConsumption += processingEnergy;
        
        // Store processed data in cache
        dataCache.put(deviceId, processedData);
        
        // Determine if data needs to be forwarded to fog node
        // In this implementation, we forward data if it exceeds a certain size threshold
        // or if it requires further processing
        if (data.length > 1024 * 100) { // More than 100KB
            forwardToFogNode(deviceId, processedData);
        }
    }
    
    /**
     * Forwards processed data to the parent fog node
     * 
     * @param deviceId ID of the original device
     * @param data The processed data to forward
     */
    private void forwardToFogNode(String deviceId, byte[] data) {
        // Apply encryption before forwarding if security is enabled
        if (securityManager.isSecurityEnabled()) {
            double beforeEncryption = System.currentTimeMillis();
            byte[] encryptedData = securityManager.encryptData(data, SecurityLevel.HIGH);
            double encryptionTime = System.currentTimeMillis() - beforeEncryption;
            
            // Update security overhead
            this.securityOverhead += encryptionTime;
            
            // Calculate energy consumption for encryption
            double encryptionEnergy = 0.001 * (data.length / 1024.0) * SecurityLevel.HIGH.getFactor();
            this.energyConsumption += encryptionEnergy;
            
            // Forward encrypted data to fog node
            parentFog.receiveData(nodeId, deviceId, encryptedData, true);
        } else {
            // Forward unencrypted data
            parentFog.receiveData(nodeId, deviceId, data, false);
        }
        
        // Calculate transmission energy
        double transmissionEnergy = 0.01 * (data.length / 1024.0); // 0.01 mJ/KB
        this.energyConsumption += transmissionEnergy;
    }
    
    /**
     * Performs data aggregation from all connected devices
     * 
     * @return Aggregated data size in KB
     */
    public double aggregateData() {
        double totalDataSize = 0.0;
        
        for (Map.Entry<String, byte[]> entry : dataCache.entrySet()) {
            totalDataSize += entry.getValue().length / 1024.0; // Convert bytes to KB
        }
        
        return totalDataSize;
    }
    
    /**
     * Clears the data cache
     */
    public void clearCache() {
        dataCache.clear();
    }
    
    // Getters
    public String getNodeId() {
        return nodeId;
    }
    
    public double getProcessingTime() {
        return processingTime;
    }
    
    public double getEnergyConsumption() {
        return energyConsumption;
    }
    
    public double getSecurityOverhead() {
        return securityOverhead;
    }
    
    public int getConnectedDevicesCount() {
        return connectedDevices.size();
    }
    
    public double getProcessingCapacity() {
        return processingCapacity;
    }
    
    public double getStorageCapacity() {
        return storageCapacity;
    }
    
    public double getAvailableBandwidth() {
        return availableBandwidth;
    }
}
