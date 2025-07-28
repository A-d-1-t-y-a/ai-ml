package com.nci.fogedge.devices;

import com.nci.fogedge.model.SimulationConfig;
import com.nci.fogedge.model.SimulationResults;
import com.nci.fogedge.security.SecurityManager;

import java.util.*;

/**
 * Manages all devices in the simulation, including creation, updates, and tracking.
 * This class is responsible for maintaining the state of all devices.
 */
public class DeviceManager {
    private SimulationConfig config;
    private SimulationResults results;
    private Map<String, Device> devices;
    private Map<DeviceType, List<Device>> devicesByType;
    private Random random;
    private SecurityManager securityManager;
    
    /**
     * Constructor for DeviceManager
     * 
     * @param config Simulation configuration
     * @param results Simulation results collector
     * @param securityManager Security manager for handling device security
     */
    public DeviceManager(SimulationConfig config, SimulationResults results, SecurityManager securityManager) {
        this.config = config;
        this.results = results;
        this.securityManager = securityManager;
        this.devices = new HashMap<>();
        this.devicesByType = new HashMap<>();
        this.random = new Random();
        
        // Initialize device lists by type
        for (DeviceType type : DeviceType.values()) {
            devicesByType.put(type, new ArrayList<>());
        }
    }
    
    /**
     * Initializes the device manager and creates devices based on configuration
     */
    public void initialize() {
        // Clear existing devices
        devices.clear();
        for (DeviceType type : DeviceType.values()) {
            devicesByType.get(type).clear();
        }
        
        // Create devices based on configuration
        createIoTDevices();
        createEdgeNodes();
        createFogNodes();
        createCloudDatacenters();
        
        // Update simulation results with device counts
        updateDeviceCountStatistics();
    }
    
    /**
     * Creates IoT devices based on configuration
     */
    private void createIoTDevices() {
        int numIoTDevices = config.getNumIoTDevices();
        
        for (int i = 0; i < numIoTDevices; i++) {
            // Generate device ID
            String deviceId = "iot_" + i;
            
            // Generate random position within the simulation area
            double xPos = random.nextDouble() * config.getSimulationAreaWidth();
            double yPos = random.nextDouble() * config.getSimulationAreaHeight();
            
            // Select a random wireless type
            WirelessType wirelessType = getRandomWirelessType();
            
            // Determine if the device is mobile
            boolean isMobile = random.nextDouble() < config.getIoTMobilityRate();
            
            // Create the IoT device
            IoTDevice device = new IoTDevice(
                deviceId,
                "IoT Device " + i,
                xPos,
                yPos,
                config.getIoTDeviceCpuCapacity(),
                config.getIoTDeviceRamCapacity(),
                config.getIoTDeviceStorageCapacity(),
                config.getIoTDeviceBatteryCapacity(),
                wirelessType,
                isMobile,
                config.getIoTDeviceTaskGenerationRate(),
                config.getIoTDeviceDataGenerationRate()
            );
            
            // Add the device to the maps
            devices.put(deviceId, device);
            devicesByType.get(DeviceType.IOT_DEVICE).add(device);
        }
    }
    
    /**
     * Creates edge nodes based on configuration
     */
    private void createEdgeNodes() {
        int numEdgeNodes = config.getNumEdgeNodes();
        
        for (int i = 0; i < numEdgeNodes; i++) {
            // Generate device ID
            String deviceId = "edge_" + i;
            
            // Generate position - edge nodes should be distributed across the simulation area
            double xPos = (i % Math.sqrt(numEdgeNodes)) * (config.getSimulationAreaWidth() / Math.sqrt(numEdgeNodes));
            double yPos = (i / Math.sqrt(numEdgeNodes)) * (config.getSimulationAreaHeight() / Math.sqrt(numEdgeNodes));
            
            // Add some randomness to the position
            xPos += random.nextDouble() * 100 - 50;
            yPos += random.nextDouble() * 100 - 50;
            
            // Ensure position is within the simulation area
            xPos = Math.max(0, Math.min(xPos, config.getSimulationAreaWidth()));
            yPos = Math.max(0, Math.min(yPos, config.getSimulationAreaHeight()));
            
            // Create the edge node
            EdgeNode device = new EdgeNode(
                deviceId,
                "Edge Node " + i,
                xPos,
                yPos,
                config.getEdgeNodeCpuCapacity(),
                config.getEdgeNodeRamCapacity(),
                config.getEdgeNodeStorageCapacity(),
                config.getEdgeNodeBatteryCapacity(),
                config.getEdgeNodeNetworkBandwidth(),
                config.getEdgeNodeNetworkLatency(),
                config.getEdgeNodeSecurityLevel()
            );
            
            // Add the device to the maps
            devices.put(deviceId, device);
            devicesByType.get(DeviceType.EDGE_NODE).add(device);
        }
    }
    
    /**
     * Creates fog nodes based on configuration
     */
    private void createFogNodes() {
        int numFogNodes = config.getNumFogNodes();
        
        for (int i = 0; i < numFogNodes; i++) {
            // Generate device ID
            String deviceId = "fog_" + i;
            
            // Generate position - fog nodes should be distributed across the simulation area
            // but fewer than edge nodes
            double xPos = (i % Math.sqrt(numFogNodes)) * (config.getSimulationAreaWidth() / Math.sqrt(numFogNodes));
            double yPos = (i / Math.sqrt(numFogNodes)) * (config.getSimulationAreaHeight() / Math.sqrt(numFogNodes));
            
            // Add some randomness to the position
            xPos += random.nextDouble() * 200 - 100;
            yPos += random.nextDouble() * 200 - 100;
            
            // Ensure position is within the simulation area
            xPos = Math.max(0, Math.min(xPos, config.getSimulationAreaWidth()));
            yPos = Math.max(0, Math.min(yPos, config.getSimulationAreaHeight()));
            
            // Create the fog node
            FogNode device = new FogNode(
                deviceId,
                "Fog Node " + i,
                xPos,
                yPos,
                config.getFogNodeCpuCapacity(),
                config.getFogNodeRamCapacity(),
                config.getFogNodeStorageCapacity(),
                config.getFogNodeBatteryCapacity(),
                config.getFogNodeNetworkBandwidth(),
                config.getFogNodeNetworkLatency(),
                config.getFogNodeSecurityLevel(),
                config.getFogNodeHasBackupPower()
            );
            
            // Add the device to the maps
            devices.put(deviceId, device);
            devicesByType.get(DeviceType.FOG_NODE).add(device);
        }
    }
    
    /**
     * Creates cloud datacenters based on configuration
     */
    private void createCloudDatacenters() {
        int numCloudDatacenters = config.getNumCloudDatacenters();
        
        for (int i = 0; i < numCloudDatacenters; i++) {
            // Generate device ID
            String deviceId = "cloud_" + i;
            
            // Generate position - cloud datacenters are typically far from the edge
            // We'll place them outside the main simulation area to represent their remote nature
            double xPos = random.nextDouble() * config.getSimulationAreaWidth() * 2;
            double yPos = random.nextDouble() * config.getSimulationAreaHeight() * 2;
            
            // Create the cloud datacenter
            CloudDatacenter device = new CloudDatacenter(
                deviceId,
                "Cloud Datacenter " + i,
                xPos,
                yPos,
                config.getCloudDatacenterCpuCapacity(),
                config.getCloudDatacenterRamCapacity(),
                config.getCloudDatacenterStorageCapacity(),
                Double.MAX_VALUE, // Cloud datacenters have unlimited power
                config.getCloudDatacenterNetworkBandwidth(),
                config.getCloudDatacenterNetworkLatency(),
                config.getCloudDatacenterSecurityLevel(),
                config.getCloudDatacenterCostPerCpuCycle(),
                config.getCloudDatacenterCostPerRamMb(),
                config.getCloudDatacenterCostPerStorageGb(),
                config.getCloudDatacenterCostPerNetworkMb()
            );
            
            // Add the device to the maps
            devices.put(deviceId, device);
            devicesByType.get(DeviceType.CLOUD_DATACENTER).add(device);
        }
    }
    
    /**
     * Updates all devices for the current simulation tick
     * 
     * @param currentTick Current simulation tick
     */
    public void updateDevices(int currentTick) {
        // Update each device
        for (Device device : devices.values()) {
            // Skip inactive devices
            if (!device.isActive()) {
                continue;
            }
            
            // Update device state
            updateDeviceState(device, currentTick);
            
            // Update device position if mobile
            if (device instanceof IoTDevice && ((IoTDevice) device).isMobile()) {
                updateDevicePosition((IoTDevice) device);
            }
            
            // Update device energy
            updateDeviceEnergy(device);
            
            // Check if the device is compromised by security issues
            if (securityManager.isDeviceCompromised(device.getId())) {
                // Apply effects of being compromised
                applyCompromisedEffects(device);
            }
        }
        
        // Update device statistics
        updateDeviceStatistics();
    }
    
    /**
     * Updates the state of a device
     * 
     * @param device Device to update
     * @param currentTick Current simulation tick
     */
    private void updateDeviceState(Device device, int currentTick) {
        // Update resource utilization based on current tasks
        // This is a simplified model; in a real simulation, the utilization would depend on
        // the specific tasks being executed
        double utilizationDecay = 0.05; // 5% decay per tick
        double currentUtilization = device.getResourceUtilization();
        double newUtilization = Math.max(0, currentUtilization - utilizationDecay);
        device.updateResourceUtilization(newUtilization);
        
        // Update device-specific state
        if (device instanceof IoTDevice) {
            updateIoTDeviceState((IoTDevice) device, currentTick);
        } else if (device instanceof EdgeNode) {
            updateEdgeNodeState((EdgeNode) device, currentTick);
        } else if (device instanceof FogNode) {
            updateFogNodeState((FogNode) device, currentTick);
        } else if (device instanceof CloudDatacenter) {
            updateCloudDatacenterState((CloudDatacenter) device, currentTick);
        }
    }
    
    /**
     * Updates the state of an IoT device
     * 
     * @param device IoT device to update
     * @param currentTick Current simulation tick
     */
    private void updateIoTDeviceState(IoTDevice device, int currentTick) {
        // Generate data based on data generation rate
        double dataGenerationRate = device.getDataGenerationRate();
        if (random.nextDouble() < dataGenerationRate) {
            double dataSize = 10 + random.nextDouble() * 90; // 10-100 KB
            device.addGeneratedData(dataSize);
            results.addTotalDataGenerated(dataSize);
        }
    }
    
    /**
     * Updates the state of an edge node
     * 
     * @param device Edge node to update
     * @param currentTick Current simulation tick
     */
    private void updateEdgeNodeState(EdgeNode device, int currentTick) {
        // Update connected devices count
        int connectedDevices = countConnectedDevices(device, DeviceType.IOT_DEVICE, 200.0);
        device.setConnectedDevicesCount(connectedDevices);
    }
    
    /**
     * Updates the state of a fog node
     * 
     * @param device Fog node to update
     * @param currentTick Current simulation tick
     */
    private void updateFogNodeState(FogNode device, int currentTick) {
        // Update connected edge nodes count
        int connectedEdgeNodes = countConnectedDevices(device, DeviceType.EDGE_NODE, 5000.0);
        device.setConnectedEdgeNodesCount(connectedEdgeNodes);
        
        // Check if backup power is needed
        if (device.getEnergyLevel() < 0.1 && device.hasBackupPower()) {
            device.activateBackupPower();
        }
    }
    
    /**
     * Updates the state of a cloud datacenter
     * 
     * @param device Cloud datacenter to update
     * @param currentTick Current simulation tick
     */
    private void updateCloudDatacenterState(CloudDatacenter device, int currentTick) {
        // Update connected fog nodes count
        int connectedFogNodes = countConnectedDevices(device, DeviceType.FOG_NODE, Double.MAX_VALUE);
        device.setConnectedFogNodesCount(connectedFogNodes);
        
        // Calculate and update costs
        double cpuUsage = device.getResourceUtilization() * device.getCpuCapacity() / 100.0;
        double ramUsage = device.getResourceUtilization() * device.getRamCapacity() / 100.0;
        double storageUsage = device.getResourceUtilization() * device.getStorageCapacity() / 100.0;
        double networkUsage = device.getResourceUtilization() * device.getNetworkBandwidth() / 100.0;
        
        double cpuCost = cpuUsage * device.getCostPerCpuCycle();
        double ramCost = ramUsage * device.getCostPerRamMb();
        double storageCost = storageUsage * device.getCostPerStorageGb();
        double networkCost = networkUsage * device.getCostPerNetworkMb();
        
        double totalCost = cpuCost + ramCost + storageCost + networkCost;
        device.updateCosts(cpuCost, ramCost, storageCost, networkCost);
        
        results.addTotalCloudCost(totalCost);
    }
    
    /**
     * Updates the position of a mobile device
     * 
     * @param device Mobile device to update
     */
    private void updateDevicePosition(IoTDevice device) {
        if (!device.isMobile()) {
            return;
        }
        
        // Get current position
        double xPos = device.getXPos();
        double yPos = device.getYPos();
        
        // Generate random movement
        double moveDistance = 10.0; // 10 meters per tick
        double moveAngle = random.nextDouble() * 2 * Math.PI;
        
        // Calculate new position
        double newXPos = xPos + moveDistance * Math.cos(moveAngle);
        double newYPos = yPos + moveDistance * Math.sin(moveAngle);
        
        // Ensure new position is within the simulation area
        newXPos = Math.max(0, Math.min(newXPos, config.getSimulationAreaWidth()));
        newYPos = Math.max(0, Math.min(newYPos, config.getSimulationAreaHeight()));
        
        // Update device position
        device.updatePosition(newXPos, newYPos);
    }
    
    /**
     * Updates the energy level of a device
     * 
     * @param device Device to update
     */
    private void updateDeviceEnergy(Device device) {
        // Skip cloud datacenters (unlimited energy)
        if (device instanceof CloudDatacenter) {
            return;
        }
        
        // Calculate energy consumption based on resource utilization
        double utilizationFactor = device.getResourceUtilization() / 100.0;
        double baseConsumption = 0.001; // Base consumption per tick
        double utilizationConsumption = 0.009 * utilizationFactor; // Additional consumption based on utilization
        double totalConsumption = baseConsumption + utilizationConsumption;
        
        // Adjust consumption based on device type
        if (device instanceof IoTDevice) {
            totalConsumption *= 0.5; // IoT devices consume less energy
        } else if (device instanceof EdgeNode) {
            totalConsumption *= 1.0; // Edge nodes consume normal energy
        } else if (device instanceof FogNode) {
            totalConsumption *= 2.0; // Fog nodes consume more energy
        }
        
        // Consume energy
        device.consumeEnergy(totalConsumption);
        
        // Update statistics
        results.addTotalEnergyConsumed(totalConsumption);
        
        // Check if the device has run out of energy
        if (device.getEnergyLevel() <= 0) {
            // Device has run out of energy
            handleDeviceOutOfEnergy(device);
        }
    }
    
    /**
     * Applies effects of being compromised to a device
     * 
     * @param device Compromised device
     */
    private void applyCompromisedEffects(Device device) {
        // Increase resource utilization due to malicious activities
        double utilizationIncrease = 10.0; // 10% increase
        device.updateResourceUtilization(Math.min(100, device.getResourceUtilization() + utilizationIncrease));
        
        // Increase energy consumption
        device.consumeEnergy(0.005);
        
        // Update statistics
        results.incrementCompromisedDeviceCount();
    }
    
    /**
     * Handles a device running out of energy
     * 
     * @param device Device that has run out of energy
     */
    private void handleDeviceOutOfEnergy(Device device) {
        // Check if the device has backup power (only fog nodes)
        if (device instanceof FogNode && ((FogNode) device).hasBackupPower() && !((FogNode) device).isBackupPowerActive()) {
            // Activate backup power
            ((FogNode) device).activateBackupPower();
        } else {
            // Deactivate the device
            device.setActive(false);
            
            // Update statistics
            results.incrementInactiveDeviceCount();
        }
    }
    
    /**
     * Counts the number of devices of a specific type within a given range of a device
     * 
     * @param device Center device
     * @param targetType Type of devices to count
     * @param maxDistance Maximum distance to consider
     * @return Number of devices within range
     */
    private int countConnectedDevices(Device device, DeviceType targetType, double maxDistance) {
        int count = 0;
        
        for (Device otherDevice : devicesByType.get(targetType)) {
            // Skip inactive devices
            if (!otherDevice.isActive()) {
                continue;
            }
            
            // Calculate distance
            double distance = calculateDistance(device, otherDevice);
            
            // Check if within range
            if (distance <= maxDistance) {
                count++;
            }
        }
        
        return count;
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
     * Updates device count statistics
     */
    private void updateDeviceCountStatistics() {
        results.setTotalDeviceCount(devices.size());
        results.setIoTDeviceCount(devicesByType.get(DeviceType.IOT_DEVICE).size());
        results.setEdgeNodeCount(devicesByType.get(DeviceType.EDGE_NODE).size());
        results.setFogNodeCount(devicesByType.get(DeviceType.FOG_NODE).size());
        results.setCloudDatacenterCount(devicesByType.get(DeviceType.CLOUD_DATACENTER).size());
    }
    
    /**
     * Updates device statistics
     */
    private void updateDeviceStatistics() {
        int activeDeviceCount = 0;
        int inactiveDeviceCount = 0;
        
        for (Device device : devices.values()) {
            if (device.isActive()) {
                activeDeviceCount++;
            } else {
                inactiveDeviceCount++;
            }
        }
        
        results.setActiveDeviceCount(activeDeviceCount);
        results.setInactiveDeviceCount(inactiveDeviceCount);
    }
    
    /**
     * Gets a random wireless type for IoT devices
     * 
     * @return Random wireless type
     */
    private WirelessType getRandomWirelessType() {
        WirelessType[] types = WirelessType.values();
        return types[random.nextInt(types.length)];
    }
    
    /**
     * Gets all devices
     * 
     * @return Map of all devices indexed by ID
     */
    public Map<String, Device> getDevices() {
        return devices;
    }
    
    /**
     * Gets devices by type
     * 
     * @param type Device type
     * @return List of devices of the specified type
     */
    public List<Device> getDevicesByType(DeviceType type) {
        return devicesByType.get(type);
    }
    
    /**
     * Gets a device by ID
     * 
     * @param deviceId Device ID
     * @return Device with the specified ID, or null if not found
     */
    public Device getDeviceById(String deviceId) {
        return devices.get(deviceId);
    }
    
    /**
     * Gets the nearest device of a specific type to a given device
     * 
     * @param device Source device
     * @param targetType Type of target device
     * @return Nearest device of the specified type, or null if none found
     */
    public Device getNearestDevice(Device device, DeviceType targetType) {
        Device nearestDevice = null;
        double minDistance = Double.MAX_VALUE;
        
        for (Device targetDevice : devicesByType.get(targetType)) {
            // Skip inactive devices
            if (!targetDevice.isActive()) {
                continue;
            }
            
            // Skip the source device
            if (targetDevice.getId().equals(device.getId())) {
                continue;
            }
            
            // Calculate distance
            double distance = calculateDistance(device, targetDevice);
            
            // Check if this is the nearest device so far
            if (distance < minDistance) {
                minDistance = distance;
                nearestDevice = targetDevice;
            }
        }
        
        return nearestDevice;
    }
    
    /**
     * Gets the least loaded device of a specific type
     * 
     * @param targetType Type of target device
     * @return Least loaded device of the specified type, or null if none found
     */
    public Device getLeastLoadedDevice(DeviceType targetType) {
        Device leastLoadedDevice = null;
        double minUtilization = Double.MAX_VALUE;
        
        for (Device targetDevice : devicesByType.get(targetType)) {
            // Skip inactive devices
            if (!targetDevice.isActive()) {
                continue;
            }
            
            // Check if this is the least loaded device so far
            double utilization = targetDevice.getResourceUtilization();
            if (utilization < minUtilization) {
                minUtilization = utilization;
                leastLoadedDevice = targetDevice;
            }
        }
        
        return leastLoadedDevice;
    }
    
    /**
     * Gets a random active device of a specific type
     * 
     * @param targetType Type of target device
     * @return Random active device of the specified type, or null if none found
     */
    public Device getRandomActiveDevice(DeviceType targetType) {
        List<Device> activeDevices = new ArrayList<>();
        
        for (Device device : devicesByType.get(targetType)) {
            if (device.isActive()) {
                activeDevices.add(device);
            }
        }
        
        if (activeDevices.isEmpty()) {
            return null;
        }
        
        return activeDevices.get(random.nextInt(activeDevices.size()));
    }
}
