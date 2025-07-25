package org.nci.fogedge.topology;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Class representing a Cloud Datacenter in the network topology
 * 
 * This class models a cloud datacenter that receives data from fog nodes
 * and performs final processing and storage. It implements cloud-level
 * security measures.
 * 
 * @author NCI H9FEC Student
 * @version 1.0
 */
public class CloudDatacenter {
    
    private String id;
    private Location location;
    private double processingCapacity; // in MIPS
    private double memory; // in MB
    private List<FogNode> connectedFogNodes;
    private Map<String, Double> receivedData; // Data ID -> Receive time
    private Map<String, Double> processedData; // Data ID -> Process time
    private Map<String, Object> dataStorage; // Data storage
    
    // Analytics counters
    private long totalDataReceived;
    private long totalDataProcessed;
    private double totalProcessingTime;
    
    /**
     * Constructor with parameters
     * @param id Datacenter ID
     * @param location Datacenter location
     * @param processingCapacity Processing capacity in MIPS
     * @param memory Memory in MB
     */
    public CloudDatacenter(String id, Location location, double processingCapacity, double memory) {
        this.id = id;
        this.location = location;
        this.processingCapacity = processingCapacity;
        this.memory = memory;
        this.connectedFogNodes = new ArrayList<>();
        this.receivedData = new HashMap<>();
        this.processedData = new HashMap<>();
        this.dataStorage = new HashMap<>();
        this.totalDataReceived = 0;
        this.totalDataProcessed = 0;
        this.totalProcessingTime = 0;
    }
    
    /**
     * Receive data from a fog node
     * @param dataId ID of received data
     * @param sourceFog Source fog node
     * @param currentTime Current simulation time
     * @return True if data received successfully, false otherwise
     */
    public boolean receiveData(String dataId, FogNode sourceFog, double currentTime) {
        // Record received data
        receivedData.put(dataId, currentTime);
        totalDataReceived++;
        
        // Process data
        double processingTime = calculateProcessingTime(dataId);
        double processCompleteTime = currentTime + processingTime;
        
        // Record processed data
        processedData.put(dataId, processCompleteTime);
        totalProcessingTime += processingTime;
        
        return true;
    }
    
    /**
     * Process and store data
     * @param dataId ID of data to process
     * @param currentTime Current simulation time
     * @return True if data processed successfully, false otherwise
     */
    public boolean processAndStoreData(String dataId, double currentTime) {
        // Check if data has been received
        if (!receivedData.containsKey(dataId)) {
            return false;
        }
        
        // Check if data has been processed
        if (!processedData.containsKey(dataId)) {
            return false;
        }
        
        // Check if processing is complete
        if (processedData.get(dataId) > currentTime) {
            return false; // Processing not yet complete
        }
        
        // Apply cloud analytics and storage
        Object processedData = applyCloudAnalytics(dataId);
        
        // Store data
        dataStorage.put(dataId, processedData);
        totalDataProcessed++;
        
        return true;
    }
    
    /**
     * Apply cloud analytics to data
     * @param dataId ID of data to process
     * @return Processed data object
     */
    private Object applyCloudAnalytics(String dataId) {
        // Apply cloud analytics (machine learning, big data processing, etc.)
        // In a real implementation, this would involve actual analytics
        return "CLOUD_PROCESSED(" + dataId + ")";
    }
    
    /**
     * Calculate processing time for data
     * @param dataId ID of data to process
     * @return Processing time in simulation time units
     */
    private double calculateProcessingTime(String dataId) {
        // Base processing time
        double baseTime = 0.002; // 2 ms
        
        // Adjust based on processing capacity
        baseTime = baseTime * (10000 / processingCapacity);
        
        return baseTime;
    }
    
    /**
     * Add a connected fog node
     * @param fog Fog node to connect
     */
    public void addConnectedFogNode(FogNode fog) {
        connectedFogNodes.add(fog);
    }
    
    /**
     * Get datacenter ID
     * @return Datacenter ID
     */
    public String getId() {
        return id;
    }
    
    /**
     * Get datacenter location
     * @return Datacenter location
     */
    public Location getLocation() {
        return location;
    }
    
    /**
     * Get processing capacity
     * @return Processing capacity in MIPS
     */
    public double getProcessingCapacity() {
        return processingCapacity;
    }
    
    /**
     * Get memory
     * @return Memory in MB
     */
    public double getMemory() {
        return memory;
    }
    
    /**
     * Get connected fog nodes
     * @return List of connected fog nodes
     */
    public List<FogNode> getConnectedFogNodes() {
        return connectedFogNodes;
    }
    
    /**
     * Get total data received
     * @return Total data received
     */
    public long getTotalDataReceived() {
        return totalDataReceived;
    }
    
    /**
     * Get total data processed
     * @return Total data processed
     */
    public long getTotalDataProcessed() {
        return totalDataProcessed;
    }
    
    /**
     * Get total processing time
     * @return Total processing time
     */
    public double getTotalProcessingTime() {
        return totalProcessingTime;
    }
    
    /**
     * Get average processing time
     * @return Average processing time
     */
    public double getAverageProcessingTime() {
        if (totalDataProcessed == 0) {
            return 0;
        }
        return totalProcessingTime / totalDataProcessed;
    }
    
    /**
     * Get data storage
     * @return Data storage map
     */
    public Map<String, Object> getDataStorage() {
        return dataStorage;
    }
    
    @Override
    public String toString() {
        return "CloudDatacenter{" +
                "id='" + id + '\'' +
                ", location=" + location +
                ", processingCapacity=" + processingCapacity +
                ", memory=" + memory +
                ", connectedFogNodes=" + connectedFogNodes.size() +
                '}';
    }
}
