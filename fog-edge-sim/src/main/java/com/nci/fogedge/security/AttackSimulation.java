package com.nci.fogedge.security;

/**
 * Represents a security attack simulation in the fog and edge computing environment.
 * This class models the properties and behavior of an attack.
 */
public class AttackSimulation {
    private String id;
    private String targetDeviceId;
    private AttackType type;
    private int startTick;
    private int endTick;
    private double severity;
    private boolean detected;
    
    /**
     * Constructor for AttackSimulation
     * 
     * @param id Unique identifier for the attack
     * @param targetDeviceId ID of the target device
     * @param type Type of attack
     * @param startTick Simulation tick when the attack starts
     * @param endTick Simulation tick when the attack ends
     * @param severity Severity of the attack (0-1)
     */
    public AttackSimulation(String id, String targetDeviceId, AttackType type, 
                           int startTick, int endTick, double severity) {
        this.id = id;
        this.targetDeviceId = targetDeviceId;
        this.type = type;
        this.startTick = startTick;
        this.endTick = endTick;
        this.severity = Math.max(0, Math.min(1, severity)); // Ensure between 0 and 1
        this.detected = false;
    }
    
    /**
     * Gets the attack ID
     * 
     * @return Attack ID
     */
    public String getId() {
        return id;
    }
    
    /**
     * Gets the target device ID
     * 
     * @return Target device ID
     */
    public String getTargetDeviceId() {
        return targetDeviceId;
    }
    
    /**
     * Gets the attack type
     * 
     * @return Attack type
     */
    public AttackType getType() {
        return type;
    }
    
    /**
     * Gets the start tick
     * 
     * @return Start tick
     */
    public int getStartTick() {
        return startTick;
    }
    
    /**
     * Gets the end tick
     * 
     * @return End tick
     */
    public int getEndTick() {
        return endTick;
    }
    
    /**
     * Gets the severity
     * 
     * @return Severity (0-1)
     */
    public double getSeverity() {
        return severity;
    }
    
    /**
     * Checks if the attack is detected
     * 
     * @return True if detected, false otherwise
     */
    public boolean isDetected() {
        return detected;
    }
    
    /**
     * Sets the detected status
     * 
     * @param detected Detected status
     */
    public void setDetected(boolean detected) {
        this.detected = detected;
    }
    
    /**
     * Checks if the attack is active at a given tick
     * 
     * @param currentTick Current simulation tick
     * @return True if active, false otherwise
     */
    public boolean isActive(int currentTick) {
        return currentTick >= startTick && currentTick <= endTick;
    }
    
    /**
     * Gets the remaining duration of the attack
     * 
     * @param currentTick Current simulation tick
     * @return Remaining duration in ticks
     */
    public int getRemainingDuration(int currentTick) {
        if (currentTick > endTick) {
            return 0;
        }
        return endTick - currentTick;
    }
    
    /**
     * Gets the total duration of the attack
     * 
     * @return Total duration in ticks
     */
    public int getTotalDuration() {
        return endTick - startTick;
    }
    
    @Override
    public String toString() {
        return "AttackSimulation{" +
                "id='" + id + '\'' +
                ", targetDeviceId='" + targetDeviceId + '\'' +
                ", type=" + type +
                ", startTick=" + startTick +
                ", endTick=" + endTick +
                ", severity=" + severity +
                ", detected=" + detected +
                '}';
    }
}
