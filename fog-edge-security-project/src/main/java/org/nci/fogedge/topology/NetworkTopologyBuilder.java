package org.nci.fogedge.topology;

import java.util.Random;

/**
 * Builder class for creating network topologies
 * 
 * This class provides a fluent interface for constructing network topologies
 * with various configurations of IoT devices, edge nodes, fog nodes, and cloud datacenter.
 * 
 * @author NCI H9FEC Student
 * @version 1.0
 */
public class NetworkTopologyBuilder {
    
    private NetworkTopology topology;
    private Random random;
    
    /**
     * Default constructor
     */
    public NetworkTopologyBuilder() {
        this.topology = new NetworkTopology();
        this.random = new Random(System.currentTimeMillis());
    }
    
    /**
     * Add IoT devices to the topology
     * @param count Number of IoT devices to add
     * @return This builder instance
     */
    public NetworkTopologyBuilder addIoTDevices(int count) {
        for (int i = 0; i < count; i++) {
            IoTDevice device = new IoTDevice("IoT_" + i, 
                    generateLocation(), 
                    generateDeviceType(),
                    random.nextDouble() * 0.5 + 0.1); // Data generation rate between 0.1 and 0.6
            topology.addIoTDevice(device);
        }
        return this;
    }
    
    /**
     * Add edge nodes to the topology
     * @param count Number of edge nodes to add
     * @return This builder instance
     */
    public NetworkTopologyBuilder addEdgeNodes(int count) {
        for (int i = 0; i < count; i++) {
            EdgeNode node = new EdgeNode("Edge_" + i, 
                    generateLocation(), 
                    random.nextInt(500) + 500, // Processing capacity between 500 and 1000 MIPS
                    random.nextInt(512) + 512); // Memory between 512 and 1024 MB
            topology.addEdgeNode(node);
        }
        return this;
    }
    
    /**
     * Add fog nodes to the topology
     * @param count Number of fog nodes to add
     * @return This builder instance
     */
    public NetworkTopologyBuilder addFogNodes(int count) {
        for (int i = 0; i < count; i++) {
            FogNode node = new FogNode("Fog_" + i, 
                    generateLocation(), 
                    random.nextInt(4000) + 2000, // Processing capacity between 2000 and 6000 MIPS
                    random.nextInt(4096) + 2048); // Memory between 2048 and 6144 MB
            topology.addFogNode(node);
        }
        return this;
    }
    
    /**
     * Add a cloud datacenter to the topology
     * @param count Number of cloud datacenters to add (should be 1)
     * @return This builder instance
     */
    public NetworkTopologyBuilder addCloudDatacenter(int count) {
        if (count > 0) {
            CloudDatacenter datacenter = new CloudDatacenter("Cloud_0", 
                    generateLocation(), 
                    random.nextInt(10000) + 15000, // Processing capacity between 15000 and 25000 MIPS
                    random.nextInt(16384) + 16384); // Memory between 16GB and 32GB
            topology.setCloudDatacenter(datacenter);
        }
        return this;
    }
    
    /**
     * Build the network topology
     * @return Constructed network topology
     */
    public NetworkTopology build() {
        // Connect devices to nodes based on proximity
        connectDevicesToEdges();
        connectEdgesToFogs();
        connectFogsToCloud();
        
        return topology;
    }
    
    /**
     * Connect IoT devices to edge nodes based on proximity
     */
    private void connectDevicesToEdges() {
        for (IoTDevice device : topology.getIotDevices()) {
            // Find closest edge node
            EdgeNode closestEdge = findClosestEdgeNode(device.getLocation());
            device.setConnectedEdgeNode(closestEdge);
            closestEdge.addConnectedDevice(device);
        }
    }
    
    /**
     * Connect edge nodes to fog nodes based on proximity
     */
    private void connectEdgesToFogs() {
        for (EdgeNode edge : topology.getEdgeNodes()) {
            // Find closest fog node
            FogNode closestFog = findClosestFogNode(edge.getLocation());
            edge.setConnectedFogNode(closestFog);
            closestFog.addConnectedEdgeNode(edge);
        }
    }
    
    /**
     * Connect fog nodes to cloud datacenter
     */
    private void connectFogsToCloud() {
        CloudDatacenter cloud = topology.getCloudDatacenter();
        for (FogNode fog : topology.getFogNodes()) {
            fog.setConnectedCloud(cloud);
            cloud.addConnectedFogNode(fog);
        }
    }
    
    /**
     * Find the closest edge node to a given location
     * @param location Location to find closest edge node to
     * @return Closest edge node
     */
    private EdgeNode findClosestEdgeNode(Location location) {
        EdgeNode closest = null;
        double minDistance = Double.MAX_VALUE;
        
        for (EdgeNode edge : topology.getEdgeNodes()) {
            double distance = calculateDistance(location, edge.getLocation());
            if (distance < minDistance) {
                minDistance = distance;
                closest = edge;
            }
        }
        
        return closest;
    }
    
    /**
     * Find the closest fog node to a given location
     * @param location Location to find closest fog node to
     * @return Closest fog node
     */
    private FogNode findClosestFogNode(Location location) {
        FogNode closest = null;
        double minDistance = Double.MAX_VALUE;
        
        for (FogNode fog : topology.getFogNodes()) {
            double distance = calculateDistance(location, fog.getLocation());
            if (distance < minDistance) {
                minDistance = distance;
                closest = fog;
            }
        }
        
        return closest;
    }
    
    /**
     * Calculate distance between two locations
     * @param loc1 First location
     * @param loc2 Second location
     * @return Distance between locations
     */
    private double calculateDistance(Location loc1, Location loc2) {
        return Math.sqrt(Math.pow(loc1.getX() - loc2.getX(), 2) + 
                         Math.pow(loc1.getY() - loc2.getY(), 2));
    }
    
    /**
     * Generate a random location
     * @return Random location
     */
    private Location generateLocation() {
        return new Location(random.nextDouble() * 1000, random.nextDouble() * 1000);
    }
    
    /**
     * Generate a random IoT device type
     * @return Random device type
     */
    private String generateDeviceType() {
        String[] types = {"SENSOR", "ACTUATOR", "CAMERA", "RFID", "SMART_METER"};
        return types[random.nextInt(types.length)];
    }
}
