package org.nci.fogedge.model;

/**
 * Class to hold simulation parameters for the fog computing environment
 * 
 * This class encapsulates all configurable parameters for the simulation
 * including timing, network characteristics, and simulation options.
 * 
 * @author NCI H9FEC Student
 * @version 1.0
 */
public class SimulationParameters {
    
    private int simulationLength; // in seconds
    private boolean traceEnabled;
    private double iotDataGenerationRate; // packets per second per device
    private int iotDataSize; // in KB
    private double networkBandwidth; // in Mbps
    private double edgeProcessingCapacity; // MIPS
    private double fogProcessingCapacity; // MIPS
    private double cloudProcessingCapacity; // MIPS
    private double edgeToFogLatency; // in ms
    private double fogToCloudLatency; // in ms
    private int securityOverhead; // percentage increase in processing time
    
    // Additional parameters needed by SimulationEngine
    private double initialDataGenerationDelay = 5.0; // initial delay in seconds
    private double dataGenerationInterval = 10.0; // interval between data generation in seconds
    private boolean iotEncryptionEnabled = true; // whether IoT encryption is enabled
    private double edgeToFogForwardingProbability = 0.7; // probability of forwarding from edge to fog
    private double fogToCloudForwardingProbability = 0.3; // probability of forwarding from fog to cloud
    private boolean fogBlockchainEnabled = true; // whether fog blockchain is enabled
    
    /**
     * Default constructor with default values
     */
    public SimulationParameters() {
        this.simulationLength = 3600; // 1 hour
        this.traceEnabled = false;
        this.iotDataGenerationRate = 0.5; // 1 packet every 2 seconds per device
        this.iotDataSize = 5; // 5 KB per packet
        this.networkBandwidth = 100; // 100 Mbps
        this.edgeProcessingCapacity = 1000; // 1000 MIPS
        this.fogProcessingCapacity = 5000; // 5000 MIPS
        this.cloudProcessingCapacity = 20000; // 20000 MIPS
        this.edgeToFogLatency = 10; // 10 ms
        this.fogToCloudLatency = 100; // 100 ms
        this.securityOverhead = 15; // 15% overhead for security processing
    }
    
    // Getters and setters
    
    public int getSimulationLength() {
        return simulationLength;
    }
    
    public void setSimulationLength(int simulationLength) {
        this.simulationLength = simulationLength;
    }
    
    public boolean isTraceEnabled() {
        return traceEnabled;
    }
    
    public void setTraceEnabled(boolean traceEnabled) {
        this.traceEnabled = traceEnabled;
    }
    
    public double getIotDataGenerationRate() {
        return iotDataGenerationRate;
    }
    
    public void setIotDataGenerationRate(double iotDataGenerationRate) {
        this.iotDataGenerationRate = iotDataGenerationRate;
    }
    
    public int getIotDataSize() {
        return iotDataSize;
    }
    
    public void setIotDataSize(int iotDataSize) {
        this.iotDataSize = iotDataSize;
    }
    
    public double getNetworkBandwidth() {
        return networkBandwidth;
    }
    
    public void setNetworkBandwidth(double networkBandwidth) {
        this.networkBandwidth = networkBandwidth;
    }
    
    public double getEdgeProcessingCapacity() {
        return edgeProcessingCapacity;
    }
    
    public void setEdgeProcessingCapacity(double edgeProcessingCapacity) {
        this.edgeProcessingCapacity = edgeProcessingCapacity;
    }
    
    public double getFogProcessingCapacity() {
        return fogProcessingCapacity;
    }
    
    public void setFogProcessingCapacity(double fogProcessingCapacity) {
        this.fogProcessingCapacity = fogProcessingCapacity;
    }
    
    public double getCloudProcessingCapacity() {
        return cloudProcessingCapacity;
    }
    
    public void setCloudProcessingCapacity(double cloudProcessingCapacity) {
        this.cloudProcessingCapacity = cloudProcessingCapacity;
    }
    
    public double getEdgeToFogLatency() {
        return edgeToFogLatency;
    }
    
    public void setEdgeToFogLatency(double edgeToFogLatency) {
        this.edgeToFogLatency = edgeToFogLatency;
    }
    
    public double getFogToCloudLatency() {
        return fogToCloudLatency;
    }
    
    public void setFogToCloudLatency(double fogToCloudLatency) {
        this.fogToCloudLatency = fogToCloudLatency;
    }
    
    public int getSecurityOverhead() {
        return securityOverhead;
    }
    
    public void setSecurityOverhead(int securityOverhead) {
        this.securityOverhead = securityOverhead;
    }
    
    /**
     * Get the initial data generation delay
     * @return Initial data generation delay in seconds
     */
    public double getInitialDataGenerationDelay() {
        return initialDataGenerationDelay;
    }
    
    /**
     * Set the initial data generation delay
     * @param initialDataGenerationDelay Initial data generation delay in seconds
     */
    public void setInitialDataGenerationDelay(double initialDataGenerationDelay) {
        this.initialDataGenerationDelay = initialDataGenerationDelay;
    }
    
    /**
     * Get the data generation interval
     * @return Data generation interval in seconds
     */
    public double getDataGenerationInterval() {
        return dataGenerationInterval;
    }
    
    /**
     * Set the data generation interval
     * @param dataGenerationInterval Data generation interval in seconds
     */
    public void setDataGenerationInterval(double dataGenerationInterval) {
        this.dataGenerationInterval = dataGenerationInterval;
    }
    
    /**
     * Check if IoT encryption is enabled
     * @return True if IoT encryption is enabled, false otherwise
     */
    public boolean isIotEncryptionEnabled() {
        return iotEncryptionEnabled;
    }
    
    /**
     * Set whether IoT encryption is enabled
     * @param iotEncryptionEnabled True to enable IoT encryption, false otherwise
     */
    public void setIotEncryptionEnabled(boolean iotEncryptionEnabled) {
        this.iotEncryptionEnabled = iotEncryptionEnabled;
    }
    
    /**
     * Get the edge to fog forwarding probability
     * @return Edge to fog forwarding probability (0-1)
     */
    public double getEdgeToFogForwardingProbability() {
        return edgeToFogForwardingProbability;
    }
    
    /**
     * Set the edge to fog forwarding probability
     * @param edgeToFogForwardingProbability Edge to fog forwarding probability (0-1)
     */
    public void setEdgeToFogForwardingProbability(double edgeToFogForwardingProbability) {
        this.edgeToFogForwardingProbability = edgeToFogForwardingProbability;
    }
    
    /**
     * Get the fog to cloud forwarding probability
     * @return Fog to cloud forwarding probability (0-1)
     */
    public double getFogToCloudForwardingProbability() {
        return fogToCloudForwardingProbability;
    }
    
    /**
     * Set the fog to cloud forwarding probability
     * @param fogToCloudForwardingProbability Fog to cloud forwarding probability (0-1)
     */
    public void setFogToCloudForwardingProbability(double fogToCloudForwardingProbability) {
        this.fogToCloudForwardingProbability = fogToCloudForwardingProbability;
    }
    
    /**
     * Check if fog blockchain is enabled
     * @return True if fog blockchain is enabled, false otherwise
     */
    public boolean isFogBlockchainEnabled() {
        return fogBlockchainEnabled;
    }
    
    /**
     * Set whether fog blockchain is enabled
     * @param fogBlockchainEnabled True to enable fog blockchain, false otherwise
     */
    public void setFogBlockchainEnabled(boolean fogBlockchainEnabled) {
        this.fogBlockchainEnabled = fogBlockchainEnabled;
    }
}
