package com.nci.fogedge.security;

/**
 * Enum representing different types of security events in the simulation.
 */
public enum SecurityEventType {
    /**
     * An attack was attempted but prevented
     */
    ATTACK_PREVENTED,
    
    /**
     * An attack was successful
     */
    ATTACK_SUCCESSFUL,
    
    /**
     * An intrusion was detected
     */
    INTRUSION_DETECTED,
    
    /**
     * A false positive detection occurred
     */
    FALSE_POSITIVE,
    
    /**
     * A device was recovered from compromise
     */
    DEVICE_RECOVERED
}
