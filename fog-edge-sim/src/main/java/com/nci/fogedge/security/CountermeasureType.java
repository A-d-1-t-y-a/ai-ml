package com.nci.fogedge.security;

/**
 * Enum representing different types of security countermeasures in the simulation.
 */
public enum CountermeasureType {
    /**
     * Traffic filtering to mitigate DDoS attacks
     */
    TRAFFIC_FILTERING,
    
    /**
     * Encryption to protect data from theft and eavesdropping
     */
    ENCRYPTION,
    
    /**
     * Secure communication protocols to prevent eavesdropping and man-in-the-middle attacks
     */
    SECURE_COMMUNICATION,
    
    /**
     * Authentication mechanisms to verify device and user identities
     */
    AUTHENTICATION,
    
    /**
     * Malware scanning to detect and remove malicious software
     */
    MALWARE_SCANNING,
    
    /**
     * Physical security measures to prevent tampering
     */
    PHYSICAL_SECURITY,
    
    /**
     * Intrusion detection systems to detect and respond to attacks
     */
    INTRUSION_DETECTION
}
