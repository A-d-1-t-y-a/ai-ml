package com.nci.fogedge.network;

import com.nci.fogedge.devices.Device;

/**
 * Represents a network link between two devices in the simulation.
 */
public class NetworkLink {
    private Device source;
    private Device destination;
    private NetworkCondition condition;
    private boolean isActive;
    private double utilization; // Percentage (0-100)
    
    /**
     * Constructor for NetworkLink
     * 
     * @param source Source device
     * @param destination Destination device
     * @param condition Network condition
     */
    public NetworkLink(Device source, Device destination, NetworkCondition condition) {
        this.source = source;
        this.destination = destination;
        this.condition = condition;
        this.isActive = true;
        this.utilization = 0.0;
    }
    
    /**
     * Gets the source device of the link
     * 
     * @return Source device
     */
    public Device getSource() {
        return source;
    }
    
    /**
     * Gets the destination device of the link
     * 
     * @return Destination device
     */
    public Device getDestination() {
        return destination;
    }
    
    /**
     * Gets the network condition of the link
     * 
     * @return Network condition
     */
    public NetworkCondition getCondition() {
        return condition;
    }
    
    /**
     * Sets the network condition of the link
     * 
     * @param condition Network condition
     */
    public void setCondition(NetworkCondition condition) {
        this.condition = condition;
    }
    
    /**
     * Checks if the link is active
     * 
     * @return True if the link is active, false otherwise
     */
    public boolean isActive() {
        return isActive;
    }
    
    /**
     * Sets the active status of the link
     * 
     * @param active True if the link is active, false otherwise
     */
    public void setActive(boolean active) {
        this.isActive = active;
    }
    
    /**
     * Gets the utilization of the link
     * 
     * @return Utilization percentage (0-100)
     */
    public double getUtilization() {
        return utilization;
    }
    
    /**
     * Updates the utilization of the link
     * 
     * @param utilization Utilization percentage (0-100)
     */
    public void updateUtilization(double utilization) {
        this.utilization = Math.max(0, Math.min(100, utilization));
    }
    
    /**
     * Calculates the distance of the link
     * 
     * @return Distance in meters
     */
    public double getDistance() {
        return source.distanceTo(destination);
    }
    
    /**
     * Returns a string representation of the network link
     * 
     * @return String representation of the network link
     */
    @Override
    public String toString() {
        return "NetworkLink{" +
               "source=" + source.getId() +
               ", destination=" + destination.getId() +
               ", latency=" + condition.getLatency() +
               ", bandwidth=" + condition.getBandwidth() +
               ", packetLoss=" + condition.getPacketLoss() +
               ", isActive=" + isActive +
               ", utilization=" + utilization +
               '}';
    }
}
