package com.nci.fogedge.security;

/**
 * Represents a security countermeasure in the fog and edge computing environment.
 * This class models the properties and behavior of a security measure.
 */
public class SecurityMeasure {
    private String id;
    private String targetDeviceId;
    private String attackId;
    private CountermeasureType type;
    private int startTick;
    private int endTick;
    private double effectiveness;
    
    /**
     * Constructor for SecurityMeasure
     * 
     * @param id Unique identifier for the measure
     * @param targetDeviceId ID of the target device
     * @param attackId ID of the attack this measure is responding to
     * @param type Type of countermeasure
     * @param startTick Simulation tick when the measure starts
     * @param endTick Simulation tick when the measure ends
     * @param effectiveness Effectiveness of the measure (0-1)
     */
    public SecurityMeasure(String id, String targetDeviceId, String attackId, 
                          CountermeasureType type, int startTick, int endTick, 
                          double effectiveness) {
        this.id = id;
        this.targetDeviceId = targetDeviceId;
        this.attackId = attackId;
        this.type = type;
        this.startTick = startTick;
        this.endTick = endTick;
        this.effectiveness = Math.max(0, Math.min(1, effectiveness)); // Ensure between 0 and 1
    }
    
    /**
     * Gets the measure ID
     * 
     * @return Measure ID
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
     * Gets the attack ID
     * 
     * @return Attack ID
     */
    public String getAttackId() {
        return attackId;
    }
    
    /**
     * Gets the countermeasure type
     * 
     * @return Countermeasure type
     */
    public CountermeasureType getType() {
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
     * Gets the effectiveness
     * 
     * @return Effectiveness (0-1)
     */
    public double getEffectiveness() {
        return effectiveness;
    }
    
    /**
     * Sets the effectiveness
     * 
     * @param effectiveness Effectiveness (0-1)
     */
    public void setEffectiveness(double effectiveness) {
        this.effectiveness = Math.max(0, Math.min(1, effectiveness));
    }
    
    /**
     * Checks if the measure is active at a given tick
     * 
     * @param currentTick Current simulation tick
     * @return True if active, false otherwise
     */
    public boolean isActive(int currentTick) {
        return currentTick >= startTick && currentTick <= endTick;
    }
    
    /**
     * Gets the remaining duration of the measure
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
     * Gets the total duration of the measure
     * 
     * @return Total duration in ticks
     */
    public int getTotalDuration() {
        return endTick - startTick;
    }
    
    /**
     * Gets the resource overhead of the measure
     * 
     * @return Resource overhead (0-100)
     */
    public double getResourceOverhead() {
        // Different countermeasures have different resource overheads
        switch (type) {
            case TRAFFIC_FILTERING:
                return 10.0 * effectiveness;
                
            case ENCRYPTION:
                return 15.0 * effectiveness;
                
            case SECURE_COMMUNICATION:
                return 12.0 * effectiveness;
                
            case AUTHENTICATION:
                return 8.0 * effectiveness;
                
            case MALWARE_SCANNING:
                return 20.0 * effectiveness;
                
            case PHYSICAL_SECURITY:
                return 5.0 * effectiveness;
                
            case INTRUSION_DETECTION:
                return 10.0 * effectiveness;
                
            default:
                return 10.0 * effectiveness;
        }
    }
    
    /**
     * Gets the energy overhead of the measure
     * 
     * @return Energy overhead (0-1)
     */
    public double getEnergyOverhead() {
        // Different countermeasures have different energy overheads
        switch (type) {
            case TRAFFIC_FILTERING:
                return 0.05 * effectiveness;
                
            case ENCRYPTION:
                return 0.1 * effectiveness;
                
            case SECURE_COMMUNICATION:
                return 0.08 * effectiveness;
                
            case AUTHENTICATION:
                return 0.03 * effectiveness;
                
            case MALWARE_SCANNING:
                return 0.15 * effectiveness;
                
            case PHYSICAL_SECURITY:
                return 0.01 * effectiveness;
                
            case INTRUSION_DETECTION:
                return 0.07 * effectiveness;
                
            default:
                return 0.05 * effectiveness;
        }
    }
    
    @Override
    public String toString() {
        return "SecurityMeasure{" +
                "id='" + id + '\'' +
                ", targetDeviceId='" + targetDeviceId + '\'' +
                ", attackId='" + attackId + '\'' +
                ", type=" + type +
                ", startTick=" + startTick +
                ", endTick=" + endTick +
                ", effectiveness=" + effectiveness +
                '}';
    }
}
