package com.nci.fogedge.network;

/**
 * Represents the current condition of a network link in the simulation.
 * Network conditions can change over time due to congestion, interference, etc.
 */
public class NetworkCondition {
    private String id;
    private double baseBandwidth; // Base bandwidth in Mbps
    private double baseLatency; // Base latency in ms
    private double currentBandwidth; // Current bandwidth in Mbps
    private double currentLatency; // Current latency in ms
    private double packetLossRate; // Packet loss rate (0-1)
    private double congestionFactor; // Congestion factor (0-1)
    
    /**
     * Constructor for NetworkCondition
     * 
     * @param id Unique identifier for the condition
     * @param bandwidth Base bandwidth in Mbps
     * @param latency Base latency in ms
     * @param packetLossRate Packet loss rate (0-1)
     * @param congestionFactor Congestion factor (0-1)
     */
    public NetworkCondition(String id, double bandwidth, double latency, 
                           double packetLossRate, double congestionFactor) {
        this.id = id;
        this.baseBandwidth = bandwidth;
        this.baseLatency = latency;
        this.currentBandwidth = bandwidth;
        this.currentLatency = latency;
        this.packetLossRate = Math.max(0, Math.min(1, packetLossRate)); // Ensure between 0 and 1
        this.congestionFactor = Math.max(0, Math.min(1, congestionFactor)); // Ensure between 0 and 1
    }
    
    /**
     * Gets the condition ID
     * 
     * @return Condition ID
     */
    public String getId() {
        return id;
    }
    
    /**
     * Gets the base bandwidth
     * 
     * @return Base bandwidth in Mbps
     */
    public double getBaseBandwidth() {
        return baseBandwidth;
    }
    
    /**
     * Gets the base latency
     * 
     * @return Base latency in ms
     */
    public double getBaseLatency() {
        return baseLatency;
    }
    
    /**
     * Gets the current bandwidth
     * 
     * @return Current bandwidth in Mbps
     */
    public double getCurrentBandwidth() {
        return currentBandwidth;
    }
    
    /**
     * Sets the current bandwidth
     * 
     * @param currentBandwidth Current bandwidth in Mbps
     */
    public void setCurrentBandwidth(double currentBandwidth) {
        this.currentBandwidth = currentBandwidth;
    }
    
    /**
     * Gets the current latency
     * 
     * @return Current latency in ms
     */
    public double getCurrentLatency() {
        return currentLatency;
    }
    
    /**
     * Sets the current latency
     * 
     * @param currentLatency Current latency in ms
     */
    public void setCurrentLatency(double currentLatency) {
        this.currentLatency = currentLatency;
    }
    
    /**
     * Gets the packet loss rate
     * 
     * @return Packet loss rate (0-1)
     */
    public double getPacketLossRate() {
        return packetLossRate;
    }
    
    /**
     * Sets the packet loss rate
     * 
     * @param packetLossRate Packet loss rate (0-1)
     */
    public void setPacketLossRate(double packetLossRate) {
        this.packetLossRate = Math.max(0, Math.min(1, packetLossRate));
    }
    
    /**
     * Gets the congestion factor
     * 
     * @return Congestion factor (0-1)
     */
    public double getCongestionFactor() {
        return congestionFactor;
    }
    
    /**
     * Sets the congestion factor
     * 
     * @param congestionFactor Congestion factor (0-1)
     */
    public void setCongestionFactor(double congestionFactor) {
        this.congestionFactor = Math.max(0, Math.min(1, congestionFactor));
    }
    
    /**
     * Resets the network condition to its base values
     */
    public void reset() {
        this.currentBandwidth = this.baseBandwidth;
        this.currentLatency = this.baseLatency;
        this.packetLossRate = 0.0;
        this.congestionFactor = 0.0;
    }
    
    /**
     * Calculates the effective bandwidth considering packet loss and congestion
     * 
     * @return Effective bandwidth in Mbps
     */
    public double getEffectiveBandwidth() {
        // Apply congestion factor (higher congestion means lower effective bandwidth)
        double effectiveBandwidth = currentBandwidth * (1.0 - congestionFactor * 0.5);
        
        // Apply packet loss (higher packet loss means lower effective bandwidth due to retransmissions)
        if (packetLossRate > 0) {
            effectiveBandwidth *= (1.0 - packetLossRate);
        }
        
        return effectiveBandwidth;
    }
    
    /**
     * Calculates the effective latency considering congestion
     * 
     * @return Effective latency in ms
     */
    public double getEffectiveLatency() {
        // Apply congestion factor (higher congestion means higher effective latency)
        return currentLatency * (1.0 + congestionFactor);
    }
    
    /**
     * Calculates the transfer time for a given data size
     * 
     * @param dataSizeKB Data size in KB
     * @return Transfer time in ms
     */
    public double calculateTransferTime(double dataSizeKB) {
        // Convert data size from KB to Mb
        double dataSizeMb = dataSizeKB * 8 / 1024;
        
        // Get effective bandwidth and latency
        double effectiveBandwidth = getEffectiveBandwidth();
        double effectiveLatency = getEffectiveLatency();
        
        // Calculate transfer time in milliseconds
        double transferTime = (dataSizeMb / effectiveBandwidth) * 1000;
        
        // Add latency
        transferTime += effectiveLatency;
        
        // Apply packet loss (higher packet loss means longer transfer time due to retransmissions)
        if (packetLossRate > 0) {
            transferTime /= (1.0 - packetLossRate);
        }
        
        return transferTime;
    }
    
    @Override
    public String toString() {
        return "NetworkCondition{" +
                "id='" + id + '\'' +
                ", baseBandwidth=" + baseBandwidth +
                ", baseLatency=" + baseLatency +
                ", currentBandwidth=" + currentBandwidth +
                ", currentLatency=" + currentLatency +
                ", packetLossRate=" + packetLossRate +
                ", congestionFactor=" + congestionFactor +
                '}';
    }
}
