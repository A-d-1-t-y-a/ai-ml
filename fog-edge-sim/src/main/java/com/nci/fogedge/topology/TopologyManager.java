package com.nci.fogedge.topology;

import com.nci.fogedge.devices.Device;
import com.nci.fogedge.devices.DeviceType;
import com.nci.fogedge.devices.EdgeNode;
import com.nci.fogedge.devices.FogNode;
import com.nci.fogedge.devices.CloudDatacenter;
import com.nci.fogedge.model.SimulationConfig;
import com.nci.fogedge.network.NetworkModel;

import java.util.*;

/**
 * Manages the network topology in the simulation, including device connections and hierarchy.
 * This class is responsible for creating and maintaining the network topology.
 */
public class TopologyManager {
    private SimulationConfig config;
    private Map<String, Device> devices;
    private Map<String, List<String>> connections;
    private Map<String, TopologyNode> topologyNodes;
    private NetworkModel networkModel;
    
    /**
     * Constructor for TopologyManager
     * 
     * @param config Simulation configuration
     * @param networkModel Network model for the simulation
     */
    public TopologyManager(SimulationConfig config, NetworkModel networkModel) {
        this.config = config;
        this.networkModel = networkModel;
        this.connections = new HashMap<>();
        this.topologyNodes = new HashMap<>();
    }
    
    /**
     * Initializes the topology manager
     * 
     * @param devices Map of all devices indexed by ID
     */
    public void initialize(Map<String, Device> devices) {
        this.devices = devices;
        connections.clear();
        topologyNodes.clear();
        
        // Create topology nodes for each device
        createTopologyNodes();
        
        // Create connections between devices
        createConnections();
        
        // Create network topology in the network model
        networkModel.createNetworkTopology(devices);
    }
    
    /**
     * Creates topology nodes for each device
     */
    private void createTopologyNodes() {
        for (Device device : devices.values()) {
            // Create a topology node for the device
            TopologyNode node = new TopologyNode(
                device.getId(),
                device.getName(),
                device.getType(),
                device.getXPos(),
                device.getYPos()
            );
            
            // Add the node to the map
            topologyNodes.put(device.getId(), node);
            
            // Initialize connections list for the device
            connections.put(device.getId(), new ArrayList<>());
        }
    }
    
    /**
     * Creates connections between devices based on their types and positions
     */
    private void createConnections() {
        // Connect IoT devices to edge nodes
        connectIoTDevicesToEdgeNodes();
        
        // Connect edge nodes to fog nodes
        connectEdgeNodesToFogNodes();
        
        // Connect fog nodes to cloud datacenters
        connectFogNodesToCloudDatacenters();
        
        // Connect devices of the same type (mesh connections)
        connectMeshNetworks();
    }
    
    /**
     * Connects IoT devices to edge nodes
     */
    private void connectIoTDevicesToEdgeNodes() {
        List<Device> iotDevices = getDevicesByType(DeviceType.IOT_DEVICE);
        List<Device> edgeNodes = getDevicesByType(DeviceType.EDGE_NODE);
        
        // Skip if there are no IoT devices or edge nodes
        if (iotDevices.isEmpty() || edgeNodes.isEmpty()) {
            return;
        }
        
        for (Device iotDevice : iotDevices) {
            // Find the nearest edge node
            Device nearestEdgeNode = findNearestDevice(iotDevice, edgeNodes);
            
            if (nearestEdgeNode != null) {
                // Calculate distance
                double distance = calculateDistance(iotDevice, nearestEdgeNode);
                
                // Check if within range
                double maxRange = getMaxConnectionRange(iotDevice, nearestEdgeNode);
                if (distance <= maxRange) {
                    // Add connection
                    addConnection(iotDevice.getId(), nearestEdgeNode.getId());
                    
                    // Update edge node's connected devices count
                    if (nearestEdgeNode instanceof EdgeNode) {
                        ((EdgeNode) nearestEdgeNode).incrementConnectedDevicesCount();
                    }
                }
            }
        }
    }
    
    /**
     * Connects edge nodes to fog nodes
     */
    private void connectEdgeNodesToFogNodes() {
        List<Device> edgeNodes = getDevicesByType(DeviceType.EDGE_NODE);
        List<Device> fogNodes = getDevicesByType(DeviceType.FOG_NODE);
        
        // Skip if there are no edge nodes or fog nodes
        if (edgeNodes.isEmpty() || fogNodes.isEmpty()) {
            return;
        }
        
        for (Device edgeNode : edgeNodes) {
            // Find the nearest fog node
            Device nearestFogNode = findNearestDevice(edgeNode, fogNodes);
            
            if (nearestFogNode != null) {
                // Calculate distance
                double distance = calculateDistance(edgeNode, nearestFogNode);
                
                // Check if within range
                double maxRange = getMaxConnectionRange(edgeNode, nearestFogNode);
                if (distance <= maxRange) {
                    // Add connection
                    addConnection(edgeNode.getId(), nearestFogNode.getId());
                    
                    // Update fog node's connected edge nodes count
                    if (nearestFogNode instanceof FogNode) {
                        ((FogNode) nearestFogNode).incrementConnectedEdgeNodesCount();
                    }
                }
            }
        }
    }
    
    /**
     * Connects fog nodes to cloud datacenters
     */
    private void connectFogNodesToCloudDatacenters() {
        List<Device> fogNodes = getDevicesByType(DeviceType.FOG_NODE);
        List<Device> cloudDatacenters = getDevicesByType(DeviceType.CLOUD_DATACENTER);
        
        // Skip if there are no fog nodes or cloud datacenters
        if (fogNodes.isEmpty() || cloudDatacenters.isEmpty()) {
            return;
        }
        
        for (Device fogNode : fogNodes) {
            // Find the nearest cloud datacenter
            Device nearestCloudDatacenter = findNearestDevice(fogNode, cloudDatacenters);
            
            if (nearestCloudDatacenter != null) {
                // Add connection (cloud datacenters have unlimited range)
                addConnection(fogNode.getId(), nearestCloudDatacenter.getId());
                
                // Update cloud datacenter's connected fog nodes count
                if (nearestCloudDatacenter instanceof CloudDatacenter) {
                    ((CloudDatacenter) nearestCloudDatacenter).incrementConnectedFogNodesCount();
                }
            }
        }
    }
    
    /**
     * Connects devices of the same type (mesh networks)
     */
    private void connectMeshNetworks() {
        // Connect IoT devices in a mesh network
        connectDevicesInMesh(DeviceType.IOT_DEVICE, config.getIoTMeshNetworkEnabled());
        
        // Connect edge nodes in a mesh network
        connectDevicesInMesh(DeviceType.EDGE_NODE, config.getEdgeMeshNetworkEnabled());
        
        // Connect fog nodes in a mesh network
        connectDevicesInMesh(DeviceType.FOG_NODE, config.getFogMeshNetworkEnabled());
        
        // Connect cloud datacenters in a mesh network
        connectDevicesInMesh(DeviceType.CLOUD_DATACENTER, true); // Cloud datacenters are always fully connected
    }
    
    /**
     * Connects devices of the same type in a mesh network
     * 
     * @param deviceType Type of devices to connect
     * @param meshEnabled Whether mesh networking is enabled for this device type
     */
    private void connectDevicesInMesh(DeviceType deviceType, boolean meshEnabled) {
        if (!meshEnabled) {
            return;
        }
        
        List<Device> devices = getDevicesByType(deviceType);
        
        // Skip if there are fewer than 2 devices
        if (devices.size() < 2) {
            return;
        }
        
        for (int i = 0; i < devices.size(); i++) {
            Device device1 = devices.get(i);
            
            for (int j = i + 1; j < devices.size(); j++) {
                Device device2 = devices.get(j);
                
                // Calculate distance
                double distance = calculateDistance(device1, device2);
                
                // Check if within range
                double maxRange = getMaxConnectionRange(device1, device2);
                if (distance <= maxRange) {
                    // Add bidirectional connection
                    addConnection(device1.getId(), device2.getId());
                    addConnection(device2.getId(), device1.getId());
                }
            }
        }
    }
    
    /**
     * Adds a connection between two devices
     * 
     * @param sourceId Source device ID
     * @param targetId Target device ID
     */
    private void addConnection(String sourceId, String targetId) {
        List<String> sourceConnections = connections.get(sourceId);
        if (sourceConnections != null && !sourceConnections.contains(targetId)) {
            sourceConnections.add(targetId);
        }
    }
    
    /**
     * Gets devices by type
     * 
     * @param type Device type
     * @return List of devices of the specified type
     */
    private List<Device> getDevicesByType(DeviceType type) {
        List<Device> result = new ArrayList<>();
        
        for (Device device : devices.values()) {
            if (device.getType() == type && device.isActive()) {
                result.add(device);
            }
        }
        
        return result;
    }
    
    /**
     * Finds the nearest device from a list of devices to a source device
     * 
     * @param source Source device
     * @param targetDevices List of target devices
     * @return Nearest device, or null if the list is empty
     */
    private Device findNearestDevice(Device source, List<Device> targetDevices) {
        if (targetDevices.isEmpty()) {
            return null;
        }
        
        Device nearest = null;
        double minDistance = Double.MAX_VALUE;
        
        for (Device target : targetDevices) {
            double distance = calculateDistance(source, target);
            
            if (distance < minDistance) {
                minDistance = distance;
                nearest = target;
            }
        }
        
        return nearest;
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
     * Gets the maximum connection range between two devices
     * 
     * @param device1 First device
     * @param device2 Second device
     * @return Maximum connection range in meters
     */
    private double getMaxConnectionRange(Device device1, Device device2) {
        DeviceType type1 = device1.getType();
        DeviceType type2 = device2.getType();
        
        // IoT device to IoT device: use wireless range
        if (type1 == DeviceType.IOT_DEVICE && type2 == DeviceType.IOT_DEVICE) {
            double range1 = getIoTDeviceRange(device1);
            double range2 = getIoTDeviceRange(device2);
            return Math.min(range1, range2);
        }
        
        // IoT device to edge node: use IoT device's wireless range
        if ((type1 == DeviceType.IOT_DEVICE && type2 == DeviceType.EDGE_NODE) ||
            (type1 == DeviceType.EDGE_NODE && type2 == DeviceType.IOT_DEVICE)) {
            if (type1 == DeviceType.IOT_DEVICE) {
                return getIoTDeviceRange(device1);
            } else {
                return getIoTDeviceRange(device2);
            }
        }
        
        // Edge node to edge node: medium range
        if (type1 == DeviceType.EDGE_NODE && type2 == DeviceType.EDGE_NODE) {
            return 500.0; // 500 meters
        }
        
        // Edge node to fog node: long range
        if ((type1 == DeviceType.EDGE_NODE && type2 == DeviceType.FOG_NODE) ||
            (type1 == DeviceType.FOG_NODE && type2 == DeviceType.EDGE_NODE)) {
            return 5000.0; // 5 kilometers
        }
        
        // Fog node to fog node: very long range
        if (type1 == DeviceType.FOG_NODE && type2 == DeviceType.FOG_NODE) {
            return 20000.0; // 20 kilometers
        }
        
        // Any connection to cloud datacenter: unlimited range
        if (type1 == DeviceType.CLOUD_DATACENTER || type2 == DeviceType.CLOUD_DATACENTER) {
            return Double.MAX_VALUE;
        }
        
        // Default: no connection
        return 0.0;
    }
    
    /**
     * Gets the wireless range of an IoT device
     * 
     * @param device Device to get range for
     * @return Wireless range in meters
     */
    private double getIoTDeviceRange(Device device) {
        if (device instanceof IoTDevice) {
            return ((IoTDevice) device).getWirelessType().getRange();
        }
        return 100.0; // Default range
    }
    
    /**
     * Updates the topology based on current device states
     * 
     * @param currentTick Current simulation tick
     */
    public void updateTopology(int currentTick) {
        // Check for device mobility and update connections
        updateConnectionsForMobileDevices();
        
        // Check for device failures and update connections
        updateConnectionsForFailedDevices();
    }
    
    /**
     * Updates connections for mobile devices
     */
    private void updateConnectionsForMobileDevices() {
        // Get all IoT devices
        List<Device> iotDevices = getDevicesByType(DeviceType.IOT_DEVICE);
        List<Device> edgeNodes = getDevicesByType(DeviceType.EDGE_NODE);
        
        for (Device device : iotDevices) {
            // Skip non-mobile devices
            if (!(device instanceof IoTDevice) || !((IoTDevice) device).isMobile()) {
                continue;
            }
            
            // Clear existing connections for this device
            connections.get(device.getId()).clear();
            
            // Find the nearest edge node
            Device nearestEdgeNode = findNearestDevice(device, edgeNodes);
            
            if (nearestEdgeNode != null) {
                // Calculate distance
                double distance = calculateDistance(device, nearestEdgeNode);
                
                // Check if within range
                double maxRange = getMaxConnectionRange(device, nearestEdgeNode);
                if (distance <= maxRange) {
                    // Add connection
                    addConnection(device.getId(), nearestEdgeNode.getId());
                }
            }
            
            // Update connections to other IoT devices if mesh networking is enabled
            if (config.getIoTMeshNetworkEnabled()) {
                for (Device otherDevice : iotDevices) {
                    // Skip self
                    if (otherDevice.getId().equals(device.getId())) {
                        continue;
                    }
                    
                    // Calculate distance
                    double distance = calculateDistance(device, otherDevice);
                    
                    // Check if within range
                    double maxRange = getMaxConnectionRange(device, otherDevice);
                    if (distance <= maxRange) {
                        // Add connection
                        addConnection(device.getId(), otherDevice.getId());
                    }
                }
            }
        }
    }
    
    /**
     * Updates connections for failed devices
     */
    private void updateConnectionsForFailedDevices() {
        // Find all inactive devices
        List<String> inactiveDeviceIds = new ArrayList<>();
        
        for (Device device : devices.values()) {
            if (!device.isActive()) {
                inactiveDeviceIds.add(device.getId());
            }
        }
        
        // Remove connections to inactive devices
        for (String deviceId : connections.keySet()) {
            List<String> deviceConnections = connections.get(deviceId);
            deviceConnections.removeAll(inactiveDeviceIds);
        }
    }
    
    /**
     * Checks if two devices are connected
     * 
     * @param sourceId Source device ID
     * @param targetId Target device ID
     * @return True if connected, false otherwise
     */
    public boolean areDevicesConnected(String sourceId, String targetId) {
        List<String> sourceConnections = connections.get(sourceId);
        return sourceConnections != null && sourceConnections.contains(targetId);
    }
    
    /**
     * Gets all devices connected to a device
     * 
     * @param deviceId Device ID
     * @return List of connected device IDs
     */
    public List<String> getConnectedDevices(String deviceId) {
        return new ArrayList<>(connections.getOrDefault(deviceId, Collections.emptyList()));
    }
    
    /**
     * Gets the path between two devices
     * 
     * @param sourceId Source device ID
     * @param targetId Target device ID
     * @return List of device IDs in the path, or empty list if no path exists
     */
    public List<String> getPath(String sourceId, String targetId) {
        // Use breadth-first search to find the shortest path
        Map<String, String> previous = new HashMap<>();
        Queue<String> queue = new LinkedList<>();
        Set<String> visited = new HashSet<>();
        
        // Start from the source
        queue.add(sourceId);
        visited.add(sourceId);
        
        while (!queue.isEmpty()) {
            String current = queue.poll();
            
            // Check if we've reached the target
            if (current.equals(targetId)) {
                // Reconstruct the path
                List<String> path = new ArrayList<>();
                String step = current;
                
                while (step != null) {
                    path.add(0, step);
                    step = previous.get(step);
                }
                
                return path;
            }
            
            // Explore neighbors
            List<String> neighbors = connections.getOrDefault(current, Collections.emptyList());
            
            for (String neighbor : neighbors) {
                if (!visited.contains(neighbor)) {
                    visited.add(neighbor);
                    previous.put(neighbor, current);
                    queue.add(neighbor);
                }
            }
        }
        
        // No path found
        return Collections.emptyList();
    }
    
    /**
     * Gets all topology nodes
     * 
     * @return Map of topology nodes indexed by device ID
     */
    public Map<String, TopologyNode> getTopologyNodes() {
        return topologyNodes;
    }
    
    /**
     * Gets all connections
     * 
     * @return Map of connections indexed by source device ID
     */
    public Map<String, List<String>> getConnections() {
        return connections;
    }
}
