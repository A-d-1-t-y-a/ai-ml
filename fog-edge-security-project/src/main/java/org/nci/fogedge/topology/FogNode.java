package org.nci.fogedge.topology;

import org.cloudbus.cloudsim.Log;
import org.nci.fogedge.security.SecurityManager;
import org.nci.fogedge.util.DataProcessor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Represents a Fog Node in the fog computing architecture.
 * Fog nodes are higher-level nodes that process aggregated data from edge nodes
 * and provide cloud-like services closer to the network edge.
 */
public class FogNode {
    private String nodeId;
    private SecurityManager securityManager;
    private List<EdgeNode> connectedEdgeNodes;
    private Map<String, Map<String, byte[]>> dataStore; // edgeId -> deviceId -> data
    private double processingTime;
    private double energyConsumption;
    private double securityOverhead;
    private double processingCapacity; // MIPS
    private double storageCapacity;    // GB
    private double availableBandwidth; // Mbps
    
    /**
     * Creates a new Fog Node with specified parameters
     * 
     * @param nodeId Unique identifier for the fog node
     * @param securityManager The security manager for encryption/authentication
     */
    public FogNode(String nodeId, SecurityManager securityManager) {
        this.nodeId = nodeId;
        this.securityManager = securityManager;
        this.connectedEdgeNodes = new ArrayList<>();
        this.dataStore = new HashMap<>();
        
        // Initialize metrics
        this.processingTime = 0.0;
        this.energyConsumption = 0.0;
        this.securityOverhead = 0.0;
        
        // Set fog node capabilities
        this.processingCapacity = 10000.0; // 10000 MIPS
        this.storageCapacity = 10.0;       // 10 GB
        this.availableBandwidth = 1000.0;  // 1000 Mbps
        
        Log.printLine("Fog Node " + nodeId + " created with processing capacity of " + 
                processingCapacity + " MIPS and storage capacity of " + storageCapacity + " GB");
    }
    
    /**
     * Registers an Edge node with this fog node
     * 
     * @param edgeNode The edge node to register
     */
    public void registerEdgeNode(EdgeNode edgeNode) {
        connectedEdgeNodes.add(edgeNode);
        dataStore.put(edgeNode.getNodeId(), new HashMap<>());
        Log.printLine("Edge Node " + edgeNode.getNodeId() + " registered with Fog Node " + nodeId);
    }
    
    /**
     * Receives data from an edge node, processes it, and stores the results
     * 
     * @param edgeId ID of the sending edge node
     * @param deviceId ID of the original device
     * @param data The data received
     * @param isEncrypted Whether the data is encrypted
     */
    public void receiveData(String edgeId, String deviceId, byte[] data, boolean isEncrypted) {
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
        
        // Process data at the fog level (more intensive processing)
        double beforeProcessing = System.currentTimeMillis();
        processedData = DataProcessor.processDataAtFog(processedData);
        double processingTime = System.currentTimeMillis() - beforeProcessing;
        
        // Update processing time
        this.processingTime += processingTime;
        
        // Calculate energy consumption for processing
        // Energy model: E = k * data_size * processing_time
        double processingEnergy = 0.005 * (data.length / 1024.0) * (processingTime / 1000.0);
        this.energyConsumption += processingEnergy;
        
        // Store processed data
        if (!dataStore.containsKey(edgeId)) {
            dataStore.put(edgeId, new HashMap<>());
        }
        dataStore.get(edgeId).put(deviceId, processedData);
        
        // Perform big data analytics on aggregated data
        if (shouldPerformAnalytics()) {
            performBigDataAnalytics();
        }
    }
    
    /**
     * Determines if big data analytics should be performed based on data volume
     * 
     * @return true if analytics should be performed, false otherwise
     */
    private boolean shouldPerformAnalytics() {
        // Perform analytics when we have sufficient data
        int dataPoints = 0;
        for (Map<String, byte[]> edgeData : dataStore.values()) {
            dataPoints += edgeData.size();
        }
        
        return dataPoints > 100; // Arbitrary threshold for demonstration
    }
    
    /**
     * Performs big data analytics on the aggregated data
     */
    private void performBigDataAnalytics() {
        Log.printLine("Fog Node " + nodeId + " performing big data analytics on aggregated data");
        
        double startTime = System.currentTimeMillis();
        
        // Simulate analytics processing time
        try {
            Thread.sleep(100); // 100ms for simulation purposes
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        
        // Calculate total data size
        double totalDataSize = 0.0;
        for (Map<String, byte[]> edgeData : dataStore.values()) {
            for (byte[] data : edgeData.values()) {
                totalDataSize += data.length / 1024.0; // Convert bytes to KB
            }
        }
        
        // Update processing time
        double analyticsTime = System.currentTimeMillis() - startTime;
        this.processingTime += analyticsTime;
        
        // Calculate energy consumption for analytics
        double analyticsEnergy = 0.01 * totalDataSize * (analyticsTime / 1000.0);
        this.energyConsumption += analyticsEnergy;
        
        Log.printLine("Fog Node " + nodeId + " completed analytics on " + 
                String.format("%.2f", totalDataSize) + " KB of data in " + 
                String.format("%.2f", analyticsTime) + " ms");
        
        // Clear processed data after analytics
        for (Map<String, byte[]> edgeData : dataStore.values()) {
            edgeData.clear();
        }
    }
    
    /**
     * Calculates the total data volume stored in this fog node
     * 
     * @return Total data volume in KB
     */
    public double calculateDataVolume() {
        double totalVolume = 0.0;
        
        for (Map<String, byte[]> edgeData : dataStore.values()) {
            for (byte[] data : edgeData.values()) {
                totalVolume += data.length / 1024.0; // Convert bytes to KB
            }
        }
        
        return totalVolume;
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
    
    public int getConnectedEdgeNodesCount() {
        return connectedEdgeNodes.size();
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
