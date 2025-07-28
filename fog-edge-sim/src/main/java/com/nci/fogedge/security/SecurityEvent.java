package com.nci.fogedge.security;

import com.nci.fogedge.devices.Device;

/**
 * Represents a security event in the simulation, such as an attack or intrusion detection.
 */
public class SecurityEvent {
    private double timestamp;
    private AttackType attackType;
    private Device targetDevice;
    private SecurityEventType eventType;
    private String description;
    
    /**
     * Constructor for SecurityEvent
     * 
     * @param timestamp Time when the event occurred
     * @param attackType Type of attack (null if not applicable)
     * @param targetDevice Target device
     * @param eventType Type of security event
     * @param description Description of the event
     */
    public SecurityEvent(double timestamp, AttackType attackType, Device targetDevice,
                        SecurityEventType eventType, String description) {
        this.timestamp = timestamp;
        this.attackType = attackType;
        this.targetDevice = targetDevice;
        this.eventType = eventType;
        this.description = description;
    }
    
    /**
     * Gets the timestamp of the event
     * 
     * @return Timestamp
     */
    public double getTimestamp() {
        return timestamp;
    }
    
    /**
     * Gets the attack type
     * 
     * @return Attack type (null if not applicable)
     */
    public AttackType getAttackType() {
        return attackType;
    }
    
    /**
     * Gets the target device
     * 
     * @return Target device
     */
    public Device getTargetDevice() {
        return targetDevice;
    }
    
    /**
     * Gets the event type
     * 
     * @return Event type
     */
    public SecurityEventType getEventType() {
        return eventType;
    }
    
    /**
     * Gets the description of the event
     * 
     * @return Description
     */
    public String getDescription() {
        return description;
    }
    
    /**
     * Returns a string representation of the security event
     * 
     * @return String representation of the security event
     */
    @Override
    public String toString() {
        return String.format("[%.2f] %s: %s on %s - %s",
                           timestamp,
                           eventType,
                           attackType != null ? attackType : "N/A",
                           targetDevice.getId(),
                           description);
    }
}
