package org.nci.fogedge.topology;

/**
 * Class representing a physical location in the network topology
 * 
 * This class models a 2D location for devices and nodes in the network.
 * 
 * @author NCI H9FEC Student
 * @version 1.0
 */
public class Location {
    
    private double x;
    private double y;
    
    /**
     * Constructor with coordinates
     * @param x X coordinate
     * @param y Y coordinate
     */
    public Location(double x, double y) {
        this.x = x;
        this.y = y;
    }
    
    /**
     * Get X coordinate
     * @return X coordinate
     */
    public double getX() {
        return x;
    }
    
    /**
     * Get Y coordinate
     * @return Y coordinate
     */
    public double getY() {
        return y;
    }
    
    /**
     * Calculate distance to another location
     * @param other Other location
     * @return Distance to other location
     */
    public double distanceTo(Location other) {
        return Math.sqrt(Math.pow(this.x - other.x, 2) + Math.pow(this.y - other.y, 2));
    }
    
    @Override
    public String toString() {
        return "(" + x + ", " + y + ")";
    }
}
