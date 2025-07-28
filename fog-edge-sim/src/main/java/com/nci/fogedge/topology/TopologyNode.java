package com.nci.fogedge.topology;

import com.nci.fogedge.devices.DeviceType;

/**
 * Represents a node in the network topology.
 * This class models a device in the topology with its position and connections.
 */
public class TopologyNode {
    private String id;
    private String name;
    private DeviceType type;
    private double xPos;
    private double yPos;
    
    /**
     * Constructor for TopologyNode
     * 
     * @param id Device ID
     * @param name Device name
     * @param type Device type
     * @param xPos X position in the simulation area
     * @param yPos Y position in the simulation area
     */
    public TopologyNode(String id, String name, DeviceType type, double xPos, double yPos) {
        this.id = id;
        this.name = name;
        this.type = type;
        this.xPos = xPos;
        this.yPos = yPos;
    }
    
    /**
     * Gets the node ID
     * 
     * @return Node ID
     */
    public String getId() {
        return id;
    }
    
    /**
     * Gets the node name
     * 
     * @return Node name
     */
    public String getName() {
        return name;
    }
    
    /**
     * Gets the device type
     * 
     * @return Device type
     */
    public DeviceType getType() {
        return type;
    }
    
    /**
     * Gets the X position
     * 
     * @return X position
     */
    public double getXPos() {
        return xPos;
    }
    
    /**
     * Gets the Y position
     * 
     * @return Y position
     */
    public double getYPos() {
        return yPos;
    }
    
    /**
     * Sets the X position
     * 
     * @param xPos X position
     */
    public void setXPos(double xPos) {
        this.xPos = xPos;
    }
    
    /**
     * Sets the Y position
     * 
     * @param yPos Y position
     */
    public void setYPos(double yPos) {
        this.yPos = yPos;
    }
    
    /**
     * Updates the position
     * 
     * @param xPos X position
     * @param yPos Y position
     */
    public void updatePosition(double xPos, double yPos) {
        this.xPos = xPos;
        this.yPos = yPos;
    }
    
    /**
     * Calculates the distance to another node
     * 
     * @param other Other node
     * @return Distance in meters
     */
    public double distanceTo(TopologyNode other) {
        double dx = this.xPos - other.xPos;
        double dy = this.yPos - other.yPos;
        return Math.sqrt(dx * dx + dy * dy);
    }
    
    @Override
    public String toString() {
        return "TopologyNode{" +
                "id='" + id + '\'' +
                ", name='" + name + '\'' +
                ", type=" + type +
                ", xPos=" + xPos +
                ", yPos=" + yPos +
                '}';
    }
}
