package com.nci.fogedge.security;

/**
 * Enum representing different types of security attacks in the simulation.
 */
public enum AttackType {
    /**
     * Distributed Denial of Service attack that overwhelms devices with traffic
     */
    DDOS,
    
    /**
     * Data theft attack that attempts to steal sensitive data
     */
    DATA_THEFT,
    
    /**
     * Eavesdropping attack that intercepts communications
     */
    EAVESDROPPING,
    
    /**
     * Man-in-the-middle attack that intercepts and potentially alters communications
     */
    MAN_IN_THE_MIDDLE,
    
    /**
     * Malware attack that infects devices with malicious software
     */
    MALWARE,
    
    /**
     * Physical tampering attack that physically damages or modifies devices
     */
    PHYSICAL_TAMPERING
}
