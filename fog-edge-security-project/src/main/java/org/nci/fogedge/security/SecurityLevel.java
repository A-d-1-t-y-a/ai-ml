package org.nci.fogedge.security;

/**
 * Enum representing different security levels for the fog computing architecture.
 * Each level has a corresponding factor that affects encryption strength and overhead.
 */
public enum SecurityLevel {
    LOW(1.0),
    MEDIUM(2.0),
    HIGH(3.0);
    
    private final double factor;
    
    SecurityLevel(double factor) {
        this.factor = factor;
    }
    
    public double getFactor() {
        return factor;
    }
}
