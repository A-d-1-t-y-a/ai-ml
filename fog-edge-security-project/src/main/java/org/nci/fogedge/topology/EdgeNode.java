package org.nci.fogedge.topology;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.nci.fogedge.topology.IoTDevice.IoTData;

import java.util.ArrayList;
import java.util.List;

/**
 * Class representing an Edge Node in the fog and edge computing topology
 * Based on the paper: "An Overview of Fog Computing and Edge Computing Security and Privacy Issues"
 * (Sensors 2021, 21, 8226, https://doi.org/10.3390/s21248226)
 */
public class EdgeNode {
    private static final Logger logger = LogManager.getLogger(EdgeNode.class);
    
    private String id;
    private List<IoTDevice> connectedDevices;
    private FogNode connectedFogNode;
    private double processingCapacity; // MIPS
    private double storageCapacity; // MB
    private double energyConsumption; // mW per processing unit
    private double dataReductionRatio; // How much data is reduced during processing
    private boolean compromised;
    
    public EdgeNode(String id) {
        this.id = id;
        this.connectedDevices = new ArrayList<>();
        this.processingCapacity = 1000.0; // 1000 MIPS
        this.storageCapacity = 1024.0; // 1 GB
        this.energyConsumption = 0.1; // 0.1 mW per MIPS
        this.dataReductionRatio = 0.7; // 70% data reduction
        this.compromised = false;
    }
    
    /**
     * Connect an IoT device to this edge node
     * @param device The IoT device to connect
     */
    public void connectDevice(IoTDevice device) {
        connectedDevices.add(device);
        device.connectToEdgeNode(this);
        logger.debug("Edge node {} connected to device {}", id, device.getId());
    }
    
    /**
     * Connect this edge node to a fog node
     * @param fogNode The fog node to connect to
     */
    public void connectToFogNode(FogNode fogNode) {
        this.connectedFogNode = fogNode;
        logger.debug("Edge node {} connected to fog node {}", id, fogNode.getId());
    }
    
    /**
     * Process data received from IoT devices
     * @param data List of data objects from IoT devices
     * @return Processed data object
     */
    public Object processData(List<Object> data) {
        if (data == null || data.isEmpty()) {
            logger.debug("No data to process at edge node {}", id);
            return new EdgeData(id, 0.0, System.currentTimeMillis(), compromised);
        }
        
        // Calculate total data size
        double totalDataSize = 0.0;
        int suspiciousDataCount = 0;
        
        for (Object obj : data) {
            if (obj instanceof IoTData) {
                IoTData iotData = (IoTData) obj;
                totalDataSize += iotData.getDataSize();
                
                if (iotData.isPotentiallyCompromised()) {
                    suspiciousDataCount++;
                }
            }
        }
        
        // Apply data reduction (filtering, aggregation)
        double processedDataSize = totalDataSize * dataReductionRatio;
        
        // Calculate processing time based on data size and processing capacity
        double processingTime = (totalDataSize / processingCapacity) * 1000; // in ms
        
        // Consume energy based on processing
        double energyUsed = processingTime * energyConsumption;
        
        logger.debug("Edge node {} processed {} KB of data from {} devices, reduced to {} KB",
                id, String.format("%.2f", totalDataSize), data.size(), String.format("%.2f", processedDataSize));
        logger.debug("Processing took {} ms and consumed {} mJ of energy", 
                String.format("%.2f", processingTime), String.format("%.2f", energyUsed));
        
        if (suspiciousDataCount > 0) {
            logger.warn("Edge node {} detected {} potentially compromised data objects", id, suspiciousDataCount);
        }
        
        // Create processed data object
        return new EdgeData(id, processedDataSize, System.currentTimeMillis(), compromised || suspiciousDataCount > 0);
    }
    
    /**
     * Mark this edge node as compromised (for attack simulation)
     */
    public void compromise() {
        this.compromised = true;
        logger.warn("Edge node {} has been compromised!", id);
    }
    
    /**
     * Restore this edge node to secure state
     */
    public void restore() {
        this.compromised = false;
        logger.info("Edge node {} has been restored to secure state", id);
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
    
    public List<IoTDevice> getConnectedDevices() {
        return connectedDevices;
    }
    
    public FogNode getConnectedFogNode() {
        return connectedFogNode;
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
     * Inner class representing data processed by an Edge node
     */
    public static class EdgeData {
        private String edgeNodeId;
        private double dataSize; // in KB
        private long timestamp;
        private boolean potentiallySuspicious;
        
        public EdgeData(String edgeNodeId, double dataSize, long timestamp, boolean potentiallySuspicious) {
            this.edgeNodeId = edgeNodeId;
            this.dataSize = dataSize;
            this.timestamp = timestamp;
            this.potentiallySuspicious = potentiallySuspicious;
        }
        
        public String getEdgeNodeId() {
            return edgeNodeId;
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
            return "EdgeData{" +
                    "edgeNodeId='" + edgeNodeId + '\'' +
                    ", dataSize=" + String.format("%.2f", dataSize) + " KB" +
                    ", timestamp=" + timestamp +
                    ", potentiallySuspicious=" + potentiallySuspicious +
                    '}';
        }
    }
}
