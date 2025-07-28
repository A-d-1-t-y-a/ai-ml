package com.nci.fogedge.network;

/**
 * Represents the network conditions (latency, bandwidth, packet loss) for a network link.
 */
public class NetworkCondition {
    private double latency; // Milliseconds
    private double bandwidth; // Mbps
    private double packetLoss; // Rate (0-1)
    
    /**
     * Constructor for NetworkCondition
     * 
     * @param latency Latency in milliseconds
     * @param bandwidth Bandwidth in Mbps
     * @param packetLoss Packet loss rate (0-1)
     */
    public NetworkCondition(double latency, double bandwidth, double packetLoss) {
        this.latency = latency;
        this.bandwidth = bandwidth;
        this.packetLoss = packetLoss;
    }
    
    /**
     * Gets the latency
     * 
     * @return Latency in milliseconds
     */
    public double getLatency() {
        return latency;
    }
    
    /**
     * Sets the latency
     * 
     * @param latency Latency in milliseconds
     */
    public void setLatency(double latency) {
        this.latency = Math.max(0, latency);
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
        this.bandwidth = Math.max(0.1, bandwidth); // Minimum 0.1 Mbps
    }
    
    /**
     * Gets the packet loss rate
     * 
     * @return Packet loss rate (0-1)
     */
    public double getPacketLoss() {
        return packetLoss;
    }
    
    /**
     * Sets the packet loss rate
     * 
     * @param packetLoss Packet loss rate (0-1)
     */
    public void setPacketLoss(double packetLoss) {
        this.packetLoss = Math.max(0, Math.min(1, packetLoss));
    }
    
    /**
     * Creates a copy of this network condition
     * 
     * @return A new NetworkCondition with the same values
     */
    public NetworkCondition copy() {
        return new NetworkCondition(latency, bandwidth, packetLoss);
    }
    
    /**
     * Returns a string representation of the network condition
     * 
     * @return String representation of the network condition
     */
    @Override
    public String toString() {
        return "NetworkCondition{" +
               "latency=" + latency + "ms" +
               ", bandwidth=" + bandwidth + "Mbps" +
               ", packetLoss=" + (packetLoss * 100) + "%" +
               '}';
    }
}
