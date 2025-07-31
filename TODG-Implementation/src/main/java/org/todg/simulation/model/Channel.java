package org.todg.simulation.model;

import java.util.Random;

/**
 * Represents a communication channel in the TODG simulation.
 * Channels connect IoT devices to edge servers and have stochastic properties.
 * 
 * Based on the TODG paper: "TODG: Distributed Task Offloading With Delay 
 * Guarantees for Edge Computing" (IEEE TPDS, 2021)
 */
public class Channel {
    private int channelId;
    private int sourceDeviceId;
    private int destinationServerId;
    private double bandwidth; // in Mbps
    private double baseBandwidth; // Base bandwidth capacity in Mbps
    private double interference; // Interference level (0.0 - 1.0)
    private double reliability; // Reliability level (0.0 - 1.0)
    private boolean available;
    private Random random;
    
    // Channel dynamics parameters
    private double dynamicsInterval; // Time interval for channel dynamics in seconds
    private double lastUpdateTime; // Last time the channel was updated
    private double interferenceVariability; // How much interference can vary (0.0 - 1.0)
    
    /**
     * Constructor for creating a new Channel.
     * 
     * @param channelId The unique identifier for this channel
     * @param sourceDeviceId The ID of the source IoT device
     * @param destinationServerId The ID of the destination edge server
     * @param baseBandwidth The base bandwidth capacity in Mbps
     * @param initialInterference The initial interference level (0.0 - 1.0)
     * @param reliability The reliability level (0.0 - 1.0)
     * @param dynamicsInterval The time interval for channel dynamics in seconds
     * @param interferenceVariability How much interference can vary (0.0 - 1.0)
     */
    public Channel(int channelId, int sourceDeviceId, int destinationServerId,
                  double baseBandwidth, double initialInterference, double reliability,
                  double dynamicsInterval, double interferenceVariability) {
        this.channelId = channelId;
        this.sourceDeviceId = sourceDeviceId;
        this.destinationServerId = destinationServerId;
        this.baseBandwidth = baseBandwidth;
        this.bandwidth = baseBandwidth;
        this.interference = initialInterference;
        this.reliability = reliability;
        this.available = true;
        this.random = new Random();
        this.dynamicsInterval = dynamicsInterval;
        this.lastUpdateTime = 0.0;
        this.interferenceVariability = interferenceVariability;
        
        // Initialize bandwidth based on interference
        updateEffectiveBandwidth();
    }
    
    /**
     * Updates the channel conditions based on the current time.
     * This simulates the stochastic nature of wireless channels.
     * 
     * @param currentTime The current simulation time
     */
    public void updateChannel(double currentTime) {
        // Check if it's time to update the channel
        if (currentTime - lastUpdateTime >= dynamicsInterval) {
            // Update interference level with random variation
            double interferenceChange = (random.nextDouble() * 2 - 1) * interferenceVariability;
            interference = Math.max(0.0, Math.min(1.0, interference + interferenceChange));
            
            // Update effective bandwidth based on new interference
            updateEffectiveBandwidth();
            
            // Update channel availability based on reliability
            available = (random.nextDouble() <= reliability);
            
            // Update last update time
            lastUpdateTime = currentTime;
        }
    }
    
    /**
     * Updates the effective bandwidth based on the current interference level.
     */
    private void updateEffectiveBandwidth() {
        // Effective bandwidth decreases as interference increases
        bandwidth = baseBandwidth * (1.0 - interference * 0.8);
    }
    
    /**
     * Calculates the transmission time for a given data size.
     * 
     * @param dataSizeInMB The data size in megabytes
     * @return The transmission time in seconds
     */
    public double calculateTransmissionTime(double dataSizeInMB) {
        // Convert data size from MB to Mb (megabytes to megabits)
        double dataSizeInMb = dataSizeInMB * 8;
        
        // Calculate transmission time (data size / effective bandwidth)
        return dataSizeInMb / bandwidth;
    }
    
    /**
     * Simulates the transmission of a task over this channel.
     * 
     * @param task The task to transmit
     * @param currentTime The current simulation time
     * @return The transmission time in seconds, or -1 if transmission failed
     */
    public double transmitTask(Task task, double currentTime) {
        // Update channel conditions
        updateChannel(currentTime);
        
        // Check if channel is available
        if (!available) {
            return -1; // Transmission failed
        }
        
        // Calculate transmission time
        double transmissionTime = calculateTransmissionTime(task.getDataSize());
        
        // Simulate transmission success based on reliability
        boolean transmissionSuccessful = (random.nextDouble() <= reliability);
        
        return transmissionSuccessful ? transmissionTime : -1;
    }
    
    // Getters and setters
    
    public int getChannelId() {
        return channelId;
    }
    
    public int getSourceDeviceId() {
        return sourceDeviceId;
    }
    
    public int getDestinationServerId() {
        return destinationServerId;
    }
    
    public double getBandwidth() {
        return bandwidth;
    }
    
    public double getBaseBandwidth() {
        return baseBandwidth;
    }
    
    public double getInterference() {
        return interference;
    }
    
    public double getReliability() {
        return reliability;
    }
    
    public boolean isAvailable() {
        return available;
    }
    
    public void setAvailable(boolean available) {
        this.available = available;
    }
    
    @Override
    public String toString() {
        return "Channel{" +
                "channelId=" + channelId +
                ", sourceDeviceId=" + sourceDeviceId +
                ", destinationServerId=" + destinationServerId +
                ", bandwidth=" + bandwidth +
                ", interference=" + interference +
                ", reliability=" + reliability +
                ", available=" + available +
                '}';
    }
}
