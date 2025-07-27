package org.nci.fogedge.security;

/**
 * Enum representing different attack types in fog and edge computing
 * Based on the paper: "An Overview of Fog Computing and Edge Computing Security and Privacy Issues"
 * (Sensors 2021, 21, 8226, https://doi.org/10.3390/s21248226)
 */
public enum AttackType {
    // Attacks targeting IoT devices
    IOT_PHYSICAL_TAMPERING("Physical Tampering", "Physical access to IoT devices to extract data or modify functionality", SecurityLayer.IOT),
    IOT_MALWARE_INJECTION("Malware Injection", "Injecting malicious code into IoT devices", SecurityLayer.IOT),
    IOT_BATTERY_DRAINING("Battery Draining", "Forcing IoT devices to perform energy-intensive operations", SecurityLayer.IOT),
    
    // Attacks targeting edge nodes
    EDGE_DOS("Denial of Service", "Overwhelming edge nodes with excessive requests", SecurityLayer.EDGE),
    EDGE_MAN_IN_MIDDLE("Man-in-the-Middle", "Intercepting communication between IoT and edge nodes", SecurityLayer.EDGE),
    EDGE_AUTHENTICATION_BYPASS("Authentication Bypass", "Bypassing authentication mechanisms at edge nodes", SecurityLayer.EDGE),
    
    // Attacks targeting fog nodes
    FOG_DATA_THEFT("Data Theft", "Unauthorized access to data stored in fog nodes", SecurityLayer.FOG),
    FOG_PRIVILEGE_ESCALATION("Privilege Escalation", "Gaining elevated access to fog resources", SecurityLayer.FOG),
    FOG_VM_ESCAPE("VM Escape", "Breaking out of a virtual machine to access the host system", SecurityLayer.FOG),
    
    // Network-level attacks
    NETWORK_EAVESDROPPING("Eavesdropping", "Passively listening to network traffic", SecurityLayer.NETWORK),
    NETWORK_TRAFFIC_ANALYSIS("Traffic Analysis", "Analyzing network traffic patterns", SecurityLayer.NETWORK),
    NETWORK_ROUTING_ATTACK("Routing Attack", "Manipulating routing information", SecurityLayer.NETWORK);
    
    private final String name;
    private final String description;
    private final SecurityLayer targetLayer;
    
    AttackType(String name, String description, SecurityLayer targetLayer) {
        this.name = name;
        this.description = description;
        this.targetLayer = targetLayer;
    }
    
    public String getName() {
        return name;
    }
    
    public String getDescription() {
        return description;
    }
    
    public SecurityLayer getTargetLayer() {
        return targetLayer;
    }
    
    @Override
    public String toString() {
        return name + " (" + targetLayer + ")";
    }
}
