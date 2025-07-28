package com.nci.fogedge.devices;

/**
 * Enum representing the different types of devices in the simulation.
 */
public enum DeviceType {
    /**
     * IoT devices are resource-constrained end devices that generate data and tasks.
     * Examples: sensors, actuators, smartphones, wearables.
     */
    IOT_DEVICE,
    
    /**
     * Edge nodes are devices located at the edge of the network, close to IoT devices.
     * They have more resources than IoT devices but less than fog nodes.
     * Examples: gateways, routers, small servers.
     */
    EDGE_NODE,
    
    /**
     * Fog nodes are intermediate computing devices between edge and cloud.
     * They have significant computing resources and are typically located in local networks.
     * Examples: local servers, mini data centers.
     */
    FOG_NODE,
    
    /**
     * Cloud data centers are large-scale computing facilities with abundant resources.
     * They are typically located far from IoT devices and have high latency but high computing power.
     * Examples: AWS, Azure, Google Cloud data centers.
     */
    CLOUD_DATACENTER
}
