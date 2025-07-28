package com.nci.fogedge.network;

/**
 * Represents a network link between two devices in the simulation.
 * A network link has properties such as bandwidth and latency.
 */
public class NetworkLink {
    private String id;
    private String sourceDeviceId;
    private String destinationDeviceId;
    private double bandwidth; // in Mbps
    private double latency; // in ms
    
    /**
     * Constructor for NetworkLink
     * 
     * @param id Unique identifier for the link
     * @param sourceDeviceId ID of the source device
     * @param destinationDeviceId ID of the destination device
     * @param bandwidth Bandwidth in Mbps
     * @param latency Latency in ms
     */
    public NetworkLink(String id, String sourceDeviceId, String destinationDeviceId, 
                       double bandwidth, double latency) {
        this.id = id;
        this.sourceDeviceId = sourceDeviceId;
        this.destinationDeviceId = destinationDeviceId;
        this.bandwidth = bandwidth;
        this.latency = latency;
    }
    
    /**
     * Gets the link ID
     * 
     * @return Link ID
     */
    public String getId() {
        return id;
    }
    
    /**
     * Gets the source device ID
     * 
     * @return Source device ID
     */
    public String getSourceDeviceId() {
        return sourceDeviceId;
    }
    
    /**
     * Gets the destination device ID
     * 
     * @return Destination device ID
     */
    public String getDestinationDeviceId() {
        return destinationDeviceId;
    }
    
    /**
     * Gets the bandwidth
     * 
     * @return Bandwidth in Mbps
     */
    public double getBandwidth() {
        return bandwidth;
    }
    
    /**
     * Sets the bandwidth
     * 
     * @param bandwidth Bandwidth in Mbps
     */
    public void setBandwidth(double bandwidth) {
        this.bandwidth = bandwidth;
    }
    
    /**
     * Gets the latency
     * 
     * @return Latency in ms
     */
    public double getLatency() {
        return latency;
    }
    
    /**
     * Sets the latency
     * 
     * @param latency Latency in ms
     */
    public void setLatency(double latency) {
        this.latency = latency;
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
        
        // Calculate transfer time in milliseconds
        double transferTime = (dataSizeMb / bandwidth) * 1000;
        
        // Add latency
        transferTime += latency;
        
        return transferTime;
    }
    
    @Override
    public String toString() {
        return "NetworkLink{" +
                "id='" + id + '\'' +
                ", sourceDeviceId='" + sourceDeviceId + '\'' +
                ", destinationDeviceId='" + destinationDeviceId + '\'' +
                ", bandwidth=" + bandwidth +
                ", latency=" + latency +
                '}';
    }
}
