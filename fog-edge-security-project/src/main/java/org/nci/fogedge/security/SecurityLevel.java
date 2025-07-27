package org.nci.fogedge.security;

/**
 * Enum representing different security levels in the fog and edge computing simulation
 * Based on the paper: "An Overview of Fog Computing and Edge Computing Security and Privacy Issues"
 * (Sensors 2021, 21, 8226, https://doi.org/10.3390/s21248226)
 */
public enum SecurityLevel {
    LOW("Basic security measures", 0.1, 0.05),
    MEDIUM("Standard security measures", 0.3, 0.15),
    HIGH("Advanced security measures", 0.6, 0.3),
    VERY_HIGH("Comprehensive security measures", 0.9, 0.5);
    
    private final String description;
    private final double detectionRate; // Probability of detecting an attack
    private final double resourceOverhead; // Resource overhead as a fraction
    
    SecurityLevel(String description, double detectionRate, double resourceOverhead) {
        this.description = description;
        this.detectionRate = detectionRate;
        this.resourceOverhead = resourceOverhead;
    }
    
    public String getDescription() {
        return description;
    }
    
    public double getDetectionRate() {
        return detectionRate;
    }
    
    public double getResourceOverhead() {
        return resourceOverhead;
    }
    
    @Override
    public String toString() {
        return name() + " (" + description + ")";
    }
}
