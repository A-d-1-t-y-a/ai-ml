package com.nci.fogedge.network;

import com.nci.fogedge.devices.*;
import com.nci.fogedge.core.SimulationConfig;
import com.nci.fogedge.core.SimulationResults;
import com.nci.fogedge.security.SecurityManager;

import java.util.*;

/**
 * Models the network connections and communication between devices in the simulation.
 */
public class NetworkModel {
    private SimulationConfig config;
    private SimulationResults results;
    private SecurityManager securityManager;
    
    private Map<String, NetworkLink> links; // Map of device pair IDs to network links
    private Map<DeviceType, Map<DeviceType, NetworkCondition>> defaultConditions; // Default network conditions between device types
    
    private Random random;
    private double averageNetworkLatency;
    private double averageNetworkBandwidth;
    private int totalPacketsSent;
    private int totalPacketsLost;
    
    /**
     * Constructor for NetworkModel
     * 
     * @param config Simulation configuration
     * @param results Simulation results
     * @param securityManager Security manager
     */
    public NetworkModel(SimulationConfig config, SimulationResults results, SecurityManager securityManager) {
        this.config = config;
        this.results = results;
        this.securityManager = securityManager;
        
        this.links = new HashMap<>();
        this.defaultConditions = new HashMap<>();
        this.random = new Random(config.getRandomSeed());
        
        this.averageNetworkLatency = 0.0;
        this.averageNetworkBandwidth = 0.0;
        this.totalPacketsSent = 0;
        this.totalPacketsLost = 0;
        
        initializeDefaultConditions();
    }
    
    /**
     * Initializes default network conditions between different device types
     */
    private void initializeDefaultConditions() {
        // Initialize maps for each device type
        for (DeviceType type : DeviceType.values()) {
            defaultConditions.put(type, new HashMap<>());
        }
        
        // Set default conditions between device types
        
        // IoT to Edge
        setDefaultCondition(DeviceType.IOT_DEVICE, DeviceType.EDGE_NODE,
                           10.0, 50.0, 0.01); // 10ms latency, 50Mbps bandwidth, 1% packet loss
        
        // IoT to Fog
        setDefaultCondition(DeviceType.IOT_DEVICE, DeviceType.FOG_NODE,
                           30.0, 20.0, 0.02); // 30ms latency, 20Mbps bandwidth, 2% packet loss
        
        // IoT to Cloud
        setDefaultCondition(DeviceType.IOT_DEVICE, DeviceType.CLOUD_DATACENTER,
                           100.0, 10.0, 0.05); // 100ms latency, 10Mbps bandwidth, 5% packet loss
        
        // Edge to Fog
        setDefaultCondition(DeviceType.EDGE_NODE, DeviceType.FOG_NODE,
                           15.0, 100.0, 0.005); // 15ms latency, 100Mbps bandwidth, 0.5% packet loss
        
        // Edge to Cloud
        setDefaultCondition(DeviceType.EDGE_NODE, DeviceType.CLOUD_DATACENTER,
                           50.0, 50.0, 0.01); // 50ms latency, 50Mbps bandwidth, 1% packet loss
        
        // Fog to Cloud
        setDefaultCondition(DeviceType.FOG_NODE, DeviceType.CLOUD_DATACENTER,
                           30.0, 200.0, 0.005); // 30ms latency, 200Mbps bandwidth, 0.5% packet loss
        
        // IoT to IoT
        setDefaultCondition(DeviceType.IOT_DEVICE, DeviceType.IOT_DEVICE,
                           5.0, 20.0, 0.02); // 5ms latency, 20Mbps bandwidth, 2% packet loss
        
        // Edge to Edge
        setDefaultCondition(DeviceType.EDGE_NODE, DeviceType.EDGE_NODE,
                           10.0, 100.0, 0.01); // 10ms latency, 100Mbps bandwidth, 1% packet loss
        
        // Fog to Fog
        setDefaultCondition(DeviceType.FOG_NODE, DeviceType.FOG_NODE,
                           20.0, 200.0, 0.005); // 20ms latency, 200Mbps bandwidth, 0.5% packet loss
        
        // Cloud to Cloud
        setDefaultCondition(DeviceType.CLOUD_DATACENTER, DeviceType.CLOUD_DATACENTER,
                           15.0, 1000.0, 0.001); // 15ms latency, 1000Mbps bandwidth, 0.1% packet loss
    }
    
    /**
     * Sets the default network condition between two device types
     * 
     * @param sourceType Source device type
     * @param destinationType Destination device type
     * @param latency Latency in milliseconds
     * @param bandwidth Bandwidth in Mbps
     * @param packetLoss Packet loss rate (0-1)
     */
    private void setDefaultCondition(DeviceType sourceType, DeviceType destinationType,
                                    double latency, double bandwidth, double packetLoss) {
        NetworkCondition condition = new NetworkCondition(latency, bandwidth, packetLoss);
        defaultConditions.get(sourceType).put(destinationType, condition);
        
        // Set the reverse condition as well (assuming symmetric network)
        if (sourceType != destinationType) {
            defaultConditions.get(destinationType).put(sourceType, condition);
        }
    }
    
    /**
     * Creates a network link between two devices
     * 
     * @param source Source device
     * @param destination Destination device
     * @return The created network link
     */
    public NetworkLink createLink(Device source, Device destination) {
        // Get the default condition based on device types
        NetworkCondition defaultCondition = getDefaultCondition(source.getType(), destination.getType());
        
        // Calculate distance between devices
        double distance = source.distanceTo(destination);
        
        // Adjust latency based on distance (add 0.1ms per meter)
        double adjustedLatency = defaultCondition.getLatency() + (distance * 0.1);
        
        // Adjust bandwidth based on distance (reduce by 1% per 10 meters)
        double adjustedBandwidth = defaultCondition.getBandwidth() * Math.pow(0.99, distance / 10.0);
        
        // Adjust packet loss based on distance (increase by 0.1% per 100 meters)
        double adjustedPacketLoss = defaultCondition.getPacketLoss() + (distance / 100.0 * 0.001);
        
        // Create a new network condition with the adjusted parameters
        NetworkCondition condition = new NetworkCondition(
            adjustedLatency,
            adjustedBandwidth,
            adjustedPacketLoss
        );
        
        // Create a new network link
        NetworkLink link = new NetworkLink(source, destination, condition);
        
        // Store the link
        String linkId = getLinkId(source, destination);
        links.put(linkId, link);
        
        return link;
    }
    
    /**
     * Gets or creates a network link between two devices
     * 
     * @param source Source device
     * @param destination Destination device
     * @return The network link
     */
    public NetworkLink getLink(Device source, Device destination) {
        String linkId = getLinkId(source, destination);
        
        // Check if the link already exists
        if (links.containsKey(linkId)) {
            return links.get(linkId);
        }
        
        // Create a new link if it doesn't exist
        return createLink(source, destination);
    }
    
    /**
     * Gets the default network condition between two device types
     * 
     * @param sourceType Source device type
     * @param destinationType Destination device type
     * @return The default network condition
     */
    public NetworkCondition getDefaultCondition(DeviceType sourceType, DeviceType destinationType) {
        return defaultConditions.get(sourceType).get(destinationType);
    }
    
    /**
     * Calculates the latency between two devices
     * 
     * @param source Source device
     * @param destination Destination device
     * @return Latency in milliseconds
     */
    public double calculateLatency(Device source, Device destination) {
        // Get the network link
        NetworkLink link = getLink(source, destination);
        
        // Get the base latency
        double latency = link.getCondition().getLatency();
        
        // Apply congestion effect (increase latency by up to 50% based on utilization)
        double sourceUtilization = source.getResourceUtilization() / 100.0;
        double destUtilization = destination.getResourceUtilization() / 100.0;
        double utilizationFactor = Math.max(sourceUtilization, destUtilization);
        
        latency *= (1.0 + (utilizationFactor * 0.5));
        
        // Apply security overhead if security is enabled
        if (securityManager.isEncryptionEnabled()) {
            latency *= 1.1; // 10% overhead for encryption
        }
        
        // Apply random variation (±10%)
        latency *= (0.9 + (random.nextDouble() * 0.2));
        
        // Update average latency
        updateAverageLatency(latency);
        
        return latency;
    }
    
    /**
     * Calculates the time required to transfer data between two devices
     * 
     * @param source Source device
     * @param destination Destination device
     * @param dataSize Data size in KB
     * @return Transfer time in seconds
     */
    public double calculateTransferTime(Device source, Device destination, double dataSize) {
        // Get the network link
        NetworkLink link = getLink(source, destination);
        
        // Get the bandwidth in Mbps
        double bandwidth = link.getCondition().getBandwidth();
        
        // Apply congestion effect (decrease bandwidth by up to 50% based on utilization)
        double sourceUtilization = source.getResourceUtilization() / 100.0;
        double destUtilization = destination.getResourceUtilization() / 100.0;
        double utilizationFactor = Math.max(sourceUtilization, destUtilization);
        
        bandwidth *= (1.0 - (utilizationFactor * 0.5));
        
        // Apply security overhead if encryption is enabled
        if (securityManager.isEncryptionEnabled()) {
            bandwidth *= 0.9; // 10% overhead for encryption
        }
        
        // Apply random variation (±10%)
        bandwidth *= (0.9 + (random.nextDouble() * 0.2));
        
        // Update average bandwidth
        updateAverageBandwidth(bandwidth);
        
        // Calculate transfer time in seconds
        // dataSize is in KB, bandwidth is in Mbps
        // 1 KB = 0.008 Mb (8 bits per byte)
        double transferTime = (dataSize * 0.008) / bandwidth;
        
        // Add latency (convert from ms to seconds)
        transferTime += (calculateLatency(source, destination) / 1000.0);
        
        // Simulate packet loss
        double packetLoss = link.getCondition().getPacketLoss();
        
        // Increase transfer time based on packet loss (retransmissions)
        if (packetLoss > 0) {
            // Simplified model: each 1% of packet loss increases transfer time by 2%
            transferTime *= (1.0 + (packetLoss * 2.0));
            
            // Simulate lost packets
            int packets = (int) (dataSize / 1.0); // Assume 1KB per packet
            int lostPackets = 0;
            
            for (int i = 0; i < packets; i++) {
                if (random.nextDouble() < packetLoss) {
                    lostPackets++;
                }
            }
            
            totalPacketsSent += packets;
            totalPacketsLost += lostPackets;
        }
        
        return transferTime;
    }
    
    /**
     * Updates the network conditions based on the current simulation state
     * 
     * @param currentTime Current simulation time
     */
    public void updateNetworkConditions(double currentTime) {
        // Update network conditions based on time of day (simulate daily patterns)
        double timeOfDay = currentTime % 86400; // Seconds in a day
        double hourOfDay = timeOfDay / 3600; // Hour of the day (0-24)
        
        // Peak hours: 9-12 and 14-17 (business hours)
        boolean isPeakHour = (hourOfDay >= 9 && hourOfDay <= 12) || (hourOfDay >= 14 && hourOfDay <= 17);
        
        // Update all links
        for (NetworkLink link : links.values()) {
            NetworkCondition condition = link.getCondition();
            
            // During peak hours, increase latency and decrease bandwidth
            if (isPeakHour) {
                condition.setLatency(condition.getLatency() * 1.2); // 20% higher latency
                condition.setBandwidth(condition.getBandwidth() * 0.8); // 20% lower bandwidth
            } else {
                // Reset to default values
                Device source = link.getSource();
                Device destination = link.getDestination();
                NetworkCondition defaultCondition = getDefaultCondition(source.getType(), destination.getType());
                
                condition.setLatency(defaultCondition.getLatency());
                condition.setBandwidth(defaultCondition.getBandwidth());
                condition.setPacketLoss(defaultCondition.getPacketLoss());
            }
            
            // Apply random fluctuations
            condition.setLatency(condition.getLatency() * (0.9 + (random.nextDouble() * 0.2))); // ±10%
            condition.setBandwidth(condition.getBandwidth() * (0.9 + (random.nextDouble() * 0.2))); // ±10%
            condition.setPacketLoss(condition.getPacketLoss() * (0.8 + (random.nextDouble() * 0.4))); // ±20%
        }
        
        // Update simulation results
        results.setAverageNetworkLatency(averageNetworkLatency);
        results.setAverageNetworkBandwidth(averageNetworkBandwidth);
    }
    
    /**
     * Updates the average network latency
     * 
     * @param latency New latency value
     */
    private void updateAverageLatency(double latency) {
        if (averageNetworkLatency == 0.0) {
            averageNetworkLatency = latency;
        } else {
            // Exponential moving average with 0.9 weight for history
            averageNetworkLatency = (averageNetworkLatency * 0.9) + (latency * 0.1);
        }
    }
    
    /**
     * Updates the average network bandwidth
     * 
     * @param bandwidth New bandwidth value
     */
    private void updateAverageBandwidth(double bandwidth) {
        if (averageNetworkBandwidth == 0.0) {
            averageNetworkBandwidth = bandwidth;
        } else {
            // Exponential moving average with 0.9 weight for history
            averageNetworkBandwidth = (averageNetworkBandwidth * 0.9) + (bandwidth * 0.1);
        }
    }
    
    /**
     * Generates a unique ID for a link between two devices
     * 
     * @param device1 First device
     * @param device2 Second device
     * @return Link ID
     */
    private String getLinkId(Device device1, Device device2) {
        // Sort device IDs to ensure consistent link IDs regardless of order
        String id1 = device1.getId();
        String id2 = device2.getId();
        
        if (id1.compareTo(id2) < 0) {
            return id1 + "-" + id2;
        } else {
            return id2 + "-" + id1;
        }
    }
    
    /**
     * Gets the average network latency
     * 
     * @return Average network latency in milliseconds
     */
    public double getAverageNetworkLatency() {
        return averageNetworkLatency;
    }
    
    /**
     * Gets the average network bandwidth
     * 
     * @return Average network bandwidth in Mbps
     */
    public double getAverageNetworkBandwidth() {
        return averageNetworkBandwidth;
    }
    
    /**
     * Gets the total number of packets sent
     * 
     * @return Total packets sent
     */
    public int getTotalPacketsSent() {
        return totalPacketsSent;
    }
    
    /**
     * Gets the total number of packets lost
     * 
     * @return Total packets lost
     */
    public int getTotalPacketsLost() {
        return totalPacketsLost;
    }
    
    /**
     * Gets the packet loss rate
     * 
     * @return Packet loss rate (0-1)
     */
    public double getPacketLossRate() {
        if (totalPacketsSent == 0) {
            return 0.0;
        }
        return (double) totalPacketsLost / totalPacketsSent;
    }
    
    /**
     * Gets all network links
     * 
     * @return Map of link IDs to network links
     */
    public Map<String, NetworkLink> getLinks() {
        return new HashMap<>(links);
    }
}
