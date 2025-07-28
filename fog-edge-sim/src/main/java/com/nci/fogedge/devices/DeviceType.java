package com.nci.fogedge.devices;

/**
 * Enum representing the different types of devices in the simulation.
 */
public enum DeviceType {
    /**
     * Internet of Things device (sensor, actuator, etc.)
     */
    IOT_DEVICE,
    
    /**
     * Edge computing node (local processing)
     */
    EDGE_NODE,
    
    /**
     * Fog computing node (intermediate processing)
     */
    FOG_NODE,
    
    /**
     * Cloud datacenter (remote processing)
     */
    CLOUD_DATACENTER
}
