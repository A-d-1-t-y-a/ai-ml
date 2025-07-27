package org.nci.fogedge.topology;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.nci.fogedge.topology.EdgeNode.EdgeData;

import java.util.ArrayList;
import java.util.List;

/**
 * Class representing a Fog Node in the fog and edge computing topology
 * Based on the paper: "An Overview of Fog Computing and Edge Computing Security and Privacy Issues"
 * (Sensors 2021, 21, 8226, https://doi.org/10.3390/s21248226)
 */
public class FogNode {
    private static final Logger logger = LogManager.getLogger(FogNode.class);
    
    private String id;
    private List<EdgeNode> connectedEdgeNodes;
    private double processingCapacity; // MIPS
    private double storageCapacity; // MB
    private double energyConsumption; // mW per processing unit
    private double dataReductionRatio; // How much data is reduced during processing
    private boolean compromised;
    
    public FogNode(String id) {
        this.id = id;
        this.connectedEdgeNodes = new ArrayList<>();
        this.processingCapacity = 5000.0; // 5000 MIPS (more powerful than edge)
        this.storageCapacity = 10240.0; // 10 GB
        this.energyConsumption = 0.2; // 0.2 mW per MIPS (higher than edge)
        this.dataReductionRatio = 0.5; // 50% data reduction
        this.compromised = false;
    }
    
    /**
     * Connect an edge node to this fog node
     * @param edgeNode The edge node to connect
     */
    public void connectEdgeNode(EdgeNode edgeNode) {
        connectedEdgeNodes.add(edgeNode);
        edgeNode.connectToFogNode(this);
        logger.debug("Fog node {} connected to edge node {}", id, edgeNode.getId());
    }
    
    /**
     * Process data received from edge nodes
     * @param data List of data objects from edge nodes
     * @return Processed data object
     */
    public Object processData(List<Object> data) {
        if (data == null || data.isEmpty()) {
            logger.debug("No data to process at fog node {}", id);
            return new FogData(id, 0.0, System.currentTimeMillis(), compromised);
        }
        
        // Calculate total data size
        double totalDataSize = 0.0;
        int suspiciousDataCount = 0;
        
        for (Object obj : data) {
            if (obj instanceof EdgeData) {
                EdgeData edgeData = (EdgeData) obj;
                totalDataSize += edgeData.getDataSize();
                
                if (edgeData.isPotentiallySuspicious()) {
                    suspiciousDataCount++;
                }
            }
        }
        
        // Apply data reduction (advanced analytics, filtering)
        double processedDataSize = totalDataSize * dataReductionRatio;
        
        // Calculate processing time based on data size and processing capacity
        double processingTime = (totalDataSize / processingCapacity) * 1000; // in ms
        
        // Consume energy based on processing
        double energyUsed = processingTime * energyConsumption;
        
        logger.debug("Fog node {} processed {} KB of data from {} edge nodes, reduced to {} KB",
                id, String.format("%.2f", totalDataSize), data.size(), String.format("%.2f", processedDataSize));
        logger.debug("Processing took {} ms and consumed {} mJ of energy", 
                String.format("%.2f", processingTime), String.format("%.2f", energyUsed));
        
        if (suspiciousDataCount > 0) {
            logger.warn("Fog node {} detected {} potentially suspicious data objects", id, suspiciousDataCount);
        }
        
        // Create processed data object
        return new FogData(id, processedDataSize, System.currentTimeMillis(), compromised || suspiciousDataCount > 0);
    }
    
    /**
     * Mark this fog node as compromised (for attack simulation)
     */
    public void compromise() {
        this.compromised = true;
        logger.warn("Fog node {} has been compromised!", id);
    }
    
    /**
     * Restore this fog node to secure state
     */
    public void restore() {
        this.compromised = false;
        logger.info("Fog node {} has been restored to secure state", id);
    }
    
    /**
     * Calculate energy consumption for a processing task
     * @param dataSize Size of data to process in KB
     * @return Energy consumed in mJ
     */
    public double calculateEnergyConsumption(double dataSize) {
        double processingTime = (dataSize / processingCapacity) * 1000; // in ms
        return processingTime * energyConsumption;
    }
    
    // Getters and setters
    
    public String getId() {
        return id;
    }
    
    public List<EdgeNode> getConnectedEdgeNodes() {
        return connectedEdgeNodes;
    }
    
    public double getProcessingCapacity() {
        return processingCapacity;
    }
    
    public double getStorageCapacity() {
        return storageCapacity;
    }
    
    public double getEnergyConsumption() {
        return energyConsumption;
    }
    
    public double getDataReductionRatio() {
        return dataReductionRatio;
    }
    
    public boolean isCompromised() {
        return compromised;
    }
    
    /**
     * Inner class representing data processed by a Fog node
     */
    public static class FogData {
        private String fogNodeId;
        private double dataSize; // in KB
        private long timestamp;
        private boolean potentiallySuspicious;
        
        public FogData(String fogNodeId, double dataSize, long timestamp, boolean potentiallySuspicious) {
            this.fogNodeId = fogNodeId;
            this.dataSize = dataSize;
            this.timestamp = timestamp;
            this.potentiallySuspicious = potentiallySuspicious;
        }
        
        public String getFogNodeId() {
            return fogNodeId;
        }
        
        public double getDataSize() {
            return dataSize;
        }
        
        public long getTimestamp() {
            return timestamp;
        }
        
        public boolean isPotentiallySuspicious() {
            return potentiallySuspicious;
        }
        
        @Override
        public String toString() {
            return "FogData{" +
                    "fogNodeId='" + fogNodeId + '\'' +
                    ", dataSize=" + String.format("%.2f", dataSize) + " KB" +
                    ", timestamp=" + timestamp +
                    ", potentiallySuspicious=" + potentiallySuspicious +
                    '}';
        }
    }
}
