package org.nci.fogedge.security;

/**
 * Enum representing different security layers in the fog and edge computing architecture
 * Based on the paper: "An Overview of Fog Computing and Edge Computing Security and Privacy Issues"
 * (Sensors 2021, 21, 8226, https://doi.org/10.3390/s21248226)
 */
public enum SecurityLayer {
    IOT("IoT Device Layer"),
    EDGE("Edge Computing Layer"),
    FOG("Fog Computing Layer"),
    NETWORK("Network Communication Layer");
    
    private final String description;
    
    SecurityLayer(String description) {
        this.description = description;
    }
    
    public String getDescription() {
        return description;
    }
    
    @Override
    public String toString() {
        return description;
    }
}
