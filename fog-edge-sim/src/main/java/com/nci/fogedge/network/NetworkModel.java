package com.nci.fogedge.network;

import com.nci.fogedge.devices.*;
import com.nci.fogedge.model.SimulationConfig;
import com.nci.fogedge.model.SimulationResults;

import java.util.*;

/**
 * Models the network in the simulation, including topology, bandwidth, and latency.
 * This class is responsible for simulating communication between devices.
 */
public class NetworkModel {
    private SimulationConfig config;
    private SimulationResults results;
    private Map<String, Map<String, NetworkLink>> networkLinks; // Links between devices
    private Map<String, NetworkCondition> networkConditions; // Network conditions for each link
    private Random random;
    
    /**
     * Constructor for NetworkModel
     * 
     * @param config Simulation configuration
     * @param results Simulation results collector
     */
    public NetworkModel(SimulationConfig config, SimulationResults results) {
        this.config = config;
        this.results = results;
        this.networkLinks = new HashMap<>();
        this.networkConditions = new HashMap<>();
        this.random = new Random();
    }
    
    /**
     * Initializes the network model
     */
    public void initialize() {
        networkLinks.clear();
        networkConditions.clear();
    }
    
    /**
     * Creates the network topology based on the devices in the simulation
     * 
     * @param devices Map of all devices indexed by ID
     */
    public void createNetworkTopology(Map<String, Device> devices) {
        // Create links between devices based on their types and positions
        for (Device source : devices.values()) {
            for (Device destination : devices.values()) {
                // Skip self-links
                if (source.getId().equals(destination.getId())) {
                    continue;
                }
                
                // Determine if a link should be created based on device types and distance
                if (shouldCreateLink(source, destination)) {
                    // Create the network link
                    createNetworkLink(source, destination);
                }
            }
        }
        
        // Initialize network conditions for all links
        initializeNetworkConditions();
    }
    
    /**
     * Determines if a link should be created between two devices
     * 
     * @param source Source device
     * @param destination Destination device
     * @return True if a link should be created, false otherwise
     */
    private boolean shouldCreateLink(Device source, Device destination) {
        // Calculate distance between devices
        double distance = calculateDistance(source, destination);
        
        // Get the maximum connection distance based on device types
        double maxDistance = getMaxConnectionDistance(source, destination);
        
        // Check if the devices are within range
        if (distance > maxDistance) {
            return false;
        }
        
        // Additional rules based on device types
        DeviceType sourceType = source.getType();
        DeviceType destType = destination.getType();
        
        // IoT devices can connect to edge nodes and other IoT devices
        if (sourceType == DeviceType.IOT_DEVICE) {
            return destType == DeviceType.EDGE_NODE || destType == DeviceType.IOT_DEVICE;
        }
        
        // Edge nodes can connect to IoT devices, other edge nodes, and fog nodes
        if (sourceType == DeviceType.EDGE_NODE) {
            return destType == DeviceType.IOT_DEVICE || destType == DeviceType.EDGE_NODE || destType == DeviceType.FOG_NODE;
        }
        
        // Fog nodes can connect to edge nodes, other fog nodes, and cloud datacenters
        if (sourceType == DeviceType.FOG_NODE) {
            return destType == DeviceType.EDGE_NODE || destType == DeviceType.FOG_NODE || destType == DeviceType.CLOUD_DATACENTER;
        }
        
        // Cloud datacenters can connect to fog nodes and other cloud datacenters
        if (sourceType == DeviceType.CLOUD_DATACENTER) {
            return destType == DeviceType.FOG_NODE || destType == DeviceType.CLOUD_DATACENTER;
        }
        
        return false;
    }
    
    /**
     * Gets the maximum connection distance between two devices based on their types
     * 
     * @param source Source device
     * @param destination Destination device
     * @return Maximum connection distance in meters
     */
    private double getMaxConnectionDistance(Device source, Device destination) {
        DeviceType sourceType = source.getType();
        DeviceType destType = destination.getType();
        
        // IoT device to IoT device: short range
        if (sourceType == DeviceType.IOT_DEVICE && destType == DeviceType.IOT_DEVICE) {
            // If the source is an IoT device, use its wireless range
            if (source instanceof IoTDevice) {
                IoTDevice iotDevice = (IoTDevice) source;
                return iotDevice.getWirelessType().getRange();
            }
            return 100.0; // Default: 100 meters
        }
        
        // IoT device to edge node: medium range
        if ((sourceType == DeviceType.IOT_DEVICE && destType == DeviceType.EDGE_NODE) ||
            (sourceType == DeviceType.EDGE_NODE && destType == DeviceType.IOT_DEVICE)) {
            // If the source is an IoT device, use its wireless range
            if (source instanceof IoTDevice) {
                IoTDevice iotDevice = (IoTDevice) source;
                return iotDevice.getWirelessType().getRange();
            }
            return 200.0; // Default: 200 meters
        }
        
        // Edge node to edge node: medium range
        if (sourceType == DeviceType.EDGE_NODE && destType == DeviceType.EDGE_NODE) {
            return 500.0; // 500 meters
        }
        
        // Edge node to fog node: long range
        if ((sourceType == DeviceType.EDGE_NODE && destType == DeviceType.FOG_NODE) ||
            (sourceType == DeviceType.FOG_NODE && destType == DeviceType.EDGE_NODE)) {
            return 5000.0; // 5 kilometers
        }
        
        // Fog node to fog node: very long range
        if (sourceType == DeviceType.FOG_NODE && destType == DeviceType.FOG_NODE) {
            return 20000.0; // 20 kilometers
        }
        
        // Fog node to cloud datacenter: unlimited range (internet)
        if ((sourceType == DeviceType.FOG_NODE && destType == DeviceType.CLOUD_DATACENTER) ||
            (sourceType == DeviceType.CLOUD_DATACENTER && destType == DeviceType.FOG_NODE)) {
            return Double.MAX_VALUE; // Unlimited range
        }
        
        // Cloud datacenter to cloud datacenter: unlimited range (internet)
        if (sourceType == DeviceType.CLOUD_DATACENTER && destType == DeviceType.CLOUD_DATACENTER) {
            return Double.MAX_VALUE; // Unlimited range
        }
        
        // Default: no connection
        return 0.0;
    }
    
    /**
     * Creates a network link between two devices
     * 
     * @param source Source device
     * @param destination Destination device
     */
    private void createNetworkLink(Device source, Device destination) {
        // Create link ID
        String linkId = source.getId() + "-" + destination.getId();
        
        // Calculate bandwidth and latency based on device types and distance
        double distance = calculateDistance(source, destination);
        double bandwidth = calculateBandwidth(source, destination, distance);
        double latency = calculateLatency(source, destination, distance);
        
        // Create the network link
        NetworkLink link = new NetworkLink(linkId, source.getId(), destination.getId(), bandwidth, latency);
        
        // Add the link to the network links map
        Map<String, NetworkLink> sourceLinks = networkLinks.computeIfAbsent(source.getId(), k -> new HashMap<>());
        sourceLinks.put(destination.getId(), link);
    }
    
    /**
     * Calculates the Euclidean distance between two devices
     * 
     * @param device1 First device
     * @param device2 Second device
     * @return Distance between the devices in meters
     */
    private double calculateDistance(Device device1, Device device2) {
        double dx = device1.getXPos() - device2.getXPos();
        double dy = device1.getYPos() - device2.getYPos();
        return Math.sqrt(dx * dx + dy * dy);
    }
    
    /**
     * Calculates the bandwidth between two devices based on their types and distance
     * 
     * @param source Source device
     * @param destination Destination device
     * @param distance Distance between the devices in meters
     * @return Bandwidth in Mbps
     */
    private double calculateBandwidth(Device source, Device destination, double distance) {
        DeviceType sourceType = source.getType();
        DeviceType destType = destination.getType();
        
        // Base bandwidth based on device types
        double baseBandwidth = getBaseBandwidth(sourceType, destType);
        
        // Adjust bandwidth based on distance
        // The further the distance, the lower the bandwidth (inverse square law)
        double distanceFactor = 1.0;
        if (distance > 0) {
            distanceFactor = 1.0 / (1.0 + Math.log10(distance / 100.0 + 1.0));
        }
        
        // Calculate final bandwidth
        double bandwidth = baseBandwidth * distanceFactor;
        
        // Apply bandwidth variation based on configuration
        double variation = config.getNetworkBandwidthVariation();
        if (variation > 0) {
            double randomFactor = 1.0 - variation + random.nextDouble() * variation * 2;
            bandwidth *= randomFactor;
        }
        
        return Math.max(0.1, bandwidth); // Minimum bandwidth: 0.1 Mbps
    }
    
    /**
     * Gets the base bandwidth between two device types
     * 
     * @param sourceType Source device type
     * @param destType Destination device type
     * @return Base bandwidth in Mbps
     */
    private double getBaseBandwidth(DeviceType sourceType, DeviceType destType) {
        // IoT device to IoT device: low bandwidth
        if (sourceType == DeviceType.IOT_DEVICE && destType == DeviceType.IOT_DEVICE) {
            return 10.0; // 10 Mbps
        }
        
        // IoT device to edge node: medium bandwidth
        if ((sourceType == DeviceType.IOT_DEVICE && destType == DeviceType.EDGE_NODE) ||
            (sourceType == DeviceType.EDGE_NODE && destType == DeviceType.IOT_DEVICE)) {
            return 50.0; // 50 Mbps
        }
        
        // Edge node to edge node: high bandwidth
        if (sourceType == DeviceType.EDGE_NODE && destType == DeviceType.EDGE_NODE) {
            return 100.0; // 100 Mbps
        }
        
        // Edge node to fog node: high bandwidth
        if ((sourceType == DeviceType.EDGE_NODE && destType == DeviceType.FOG_NODE) ||
            (sourceType == DeviceType.FOG_NODE && destType == DeviceType.EDGE_NODE)) {
            return 200.0; // 200 Mbps
        }
        
        // Fog node to fog node: very high bandwidth
        if (sourceType == DeviceType.FOG_NODE && destType == DeviceType.FOG_NODE) {
            return 500.0; // 500 Mbps
        }
        
        // Fog node to cloud datacenter: high bandwidth
        if ((sourceType == DeviceType.FOG_NODE && destType == DeviceType.CLOUD_DATACENTER) ||
            (sourceType == DeviceType.CLOUD_DATACENTER && destType == DeviceType.FOG_NODE)) {
            return 1000.0; // 1 Gbps
        }
        
        // Cloud datacenter to cloud datacenter: extremely high bandwidth
        if (sourceType == DeviceType.CLOUD_DATACENTER && destType == DeviceType.CLOUD_DATACENTER) {
            return 10000.0; // 10 Gbps
        }
        
        // Default: low bandwidth
        return 1.0; // 1 Mbps
    }
    
    /**
     * Calculates the latency between two devices based on their types and distance
     * 
     * @param source Source device
     * @param destination Destination device
     * @param distance Distance between the devices in meters
     * @return Latency in milliseconds
     */
    private double calculateLatency(Device source, Device destination, double distance) {
        DeviceType sourceType = source.getType();
        DeviceType destType = destination.getType();
        
        // Base latency based on device types
        double baseLatency = getBaseLatency(sourceType, destType);
        
        // Adjust latency based on distance
        // The further the distance, the higher the latency (linear relationship)
        double distanceFactor = distance / 1000.0; // Convert to kilometers
        
        // Calculate propagation delay (speed of light in fiber: ~200,000 km/s)
        double propagationDelay = distance / 200000.0 * 1000.0; // Convert to milliseconds
        
        // Calculate final latency
        double latency = baseLatency + distanceFactor + propagationDelay;
        
        // Apply latency variation based on configuration
        double variation = config.getNetworkLatencyVariation();
        if (variation > 0) {
            double randomFactor = 1.0 - variation + random.nextDouble() * variation * 2;
            latency *= randomFactor;
        }
        
        return Math.max(0.1, latency); // Minimum latency: 0.1 ms
    }
    
    /**
     * Gets the base latency between two device types
     * 
     * @param sourceType Source device type
     * @param destType Destination device type
     * @return Base latency in milliseconds
     */
    private double getBaseLatency(DeviceType sourceType, DeviceType destType) {
        // IoT device to IoT device: low latency
        if (sourceType == DeviceType.IOT_DEVICE && destType == DeviceType.IOT_DEVICE) {
            return 5.0; // 5 ms
        }
        
        // IoT device to edge node: low latency
        if ((sourceType == DeviceType.IOT_DEVICE && destType == DeviceType.EDGE_NODE) ||
            (sourceType == DeviceType.EDGE_NODE && destType == DeviceType.IOT_DEVICE)) {
            return 10.0; // 10 ms
        }
        
        // Edge node to edge node: low latency
        if (sourceType == DeviceType.EDGE_NODE && destType == DeviceType.EDGE_NODE) {
            return 15.0; // 15 ms
        }
        
        // Edge node to fog node: medium latency
        if ((sourceType == DeviceType.EDGE_NODE && destType == DeviceType.FOG_NODE) ||
            (sourceType == DeviceType.FOG_NODE && destType == DeviceType.EDGE_NODE)) {
            return 30.0; // 30 ms
        }
        
        // Fog node to fog node: medium latency
        if (sourceType == DeviceType.FOG_NODE && destType == DeviceType.FOG_NODE) {
            return 50.0; // 50 ms
        }
        
        // Fog node to cloud datacenter: high latency
        if ((sourceType == DeviceType.FOG_NODE && destType == DeviceType.CLOUD_DATACENTER) ||
            (sourceType == DeviceType.CLOUD_DATACENTER && destType == DeviceType.FOG_NODE)) {
            return 100.0; // 100 ms
        }
        
        // Cloud datacenter to cloud datacenter: high latency
        if (sourceType == DeviceType.CLOUD_DATACENTER && destType == DeviceType.CLOUD_DATACENTER) {
            return 150.0; // 150 ms
        }
        
        // Default: high latency
        return 200.0; // 200 ms
    }
    
    /**
     * Initializes network conditions for all links
     */
    private void initializeNetworkConditions() {
        for (Map.Entry<String, Map<String, NetworkLink>> sourceEntry : networkLinks.entrySet()) {
            String sourceId = sourceEntry.getKey();
            
            for (Map.Entry<String, NetworkLink> destEntry : sourceEntry.getValue().entrySet()) {
                String destId = destEntry.getKey();
                NetworkLink link = destEntry.getValue();
                
                // Create network condition for the link
                String conditionId = sourceId + "-" + destId;
                NetworkCondition condition = new NetworkCondition(
                    conditionId,
                    link.getBandwidth(),
                    link.getLatency(),
                    0.0, // No packet loss initially
                    1.0  // No congestion initially
                );
                
                // Add the condition to the network conditions map
                networkConditions.put(conditionId, condition);
            }
        }
    }
    
    /**
     * Updates network conditions for all links
     * 
     * @param currentTick Current simulation tick
     */
    public void updateNetworkConditions(int currentTick) {
        // Update network conditions based on configuration and random factors
        for (NetworkCondition condition : networkConditions.values()) {
            // Apply random variations to bandwidth and latency
            updateNetworkCondition(condition);
        }
    }
    
    /**
     * Updates a network condition with random variations
     * 
     * @param condition Network condition to update
     */
    private void updateNetworkCondition(NetworkCondition condition) {
        // Get configuration parameters
        double bandwidthVariation = config.getNetworkBandwidthVariation();
        double latencyVariation = config.getNetworkLatencyVariation();
        double packetLossRate = config.getNetworkPacketLossRate();
        
        // Apply random variations to bandwidth
        if (bandwidthVariation > 0) {
            double randomFactor = 1.0 - bandwidthVariation + random.nextDouble() * bandwidthVariation * 2;
            double newBandwidth = condition.getBaseBandwidth() * randomFactor;
            condition.setCurrentBandwidth(Math.max(0.1, newBandwidth));
        }
        
        // Apply random variations to latency
        if (latencyVariation > 0) {
            double randomFactor = 1.0 - latencyVariation + random.nextDouble() * latencyVariation * 2;
            double newLatency = condition.getBaseLatency() * randomFactor;
            condition.setCurrentLatency(Math.max(0.1, newLatency));
        }
        
        // Apply random packet loss
        if (packetLossRate > 0) {
            double randomPacketLoss = random.nextDouble() * packetLossRate;
            condition.setPacketLossRate(randomPacketLoss);
        }
        
        // Apply congestion based on network usage
        // This would normally be based on actual network usage, but for simplicity,
        // we'll use a random factor
        double randomCongestion = 0.5 + random.nextDouble() * 0.5; // 0.5-1.0
        condition.setCongestionFactor(randomCongestion);
    }
    
    /**
     * Simulates data transfer between two devices
     * 
     * @param sourceId Source device ID
     * @param destId Destination device ID
     * @param dataSizeKB Data size in KB
     * @return Transfer time in milliseconds, or -1 if transfer is not possible
     */
    public double simulateDataTransfer(String sourceId, String destId, double dataSizeKB) {
        // Check if there's a direct link between the devices
        NetworkLink link = getNetworkLink(sourceId, destId);
        if (link == null) {
            return -1; // No direct link
        }
        
        // Get the network condition for the link
        String conditionId = sourceId + "-" + destId;
        NetworkCondition condition = networkConditions.get(conditionId);
        if (condition == null) {
            return -1; // No network condition
        }
        
        // Calculate transfer time based on current bandwidth, latency, and congestion
        double bandwidth = condition.getCurrentBandwidth(); // Mbps
        double latency = condition.getCurrentLatency(); // ms
        double congestion = condition.getCongestionFactor(); // 0-1
        double packetLoss = condition.getPacketLossRate(); // 0-1
        
        // Convert data size from KB to Mb
        double dataSizeMb = dataSizeKB * 8 / 1024;
        
        // Calculate base transfer time in milliseconds
        double transferTime = (dataSizeMb / bandwidth) * 1000;
        
        // Apply latency
        transferTime += latency;
        
        // Apply congestion (higher congestion means longer transfer time)
        transferTime /= (1.0 - congestion * 0.5); // Max 2x slowdown due to congestion
        
        // Apply packet loss (higher packet loss means longer transfer time due to retransmissions)
        if (packetLoss > 0) {
            transferTime /= (1.0 - packetLoss); // Retransmissions
        }
        
        return transferTime;
    }
    
    /**
     * Gets the network link between two devices
     * 
     * @param sourceId Source device ID
     * @param destId Destination device ID
     * @return NetworkLink object, or null if no link exists
     */
    public NetworkLink getNetworkLink(String sourceId, String destId) {
        Map<String, NetworkLink> sourceLinks = networkLinks.get(sourceId);
        if (sourceLinks == null) {
            return null;
        }
        return sourceLinks.get(destId);
    }
    
    /**
     * Gets the network condition between two devices
     * 
     * @param sourceId Source device ID
     * @param destId Destination device ID
     * @return NetworkCondition object, or null if no condition exists
     */
    public NetworkCondition getNetworkCondition(String sourceId, String destId) {
        String conditionId = sourceId + "-" + destId;
        return networkConditions.get(conditionId);
    }
    
    /**
     * Gets all network links
     * 
     * @return Map of network links
     */
    public Map<String, Map<String, NetworkLink>> getNetworkLinks() {
        return networkLinks;
    }
    
    /**
     * Gets all network conditions
     * 
     * @return Map of network conditions
     */
    public Map<String, NetworkCondition> getNetworkConditions() {
        return networkConditions;
    }
}
