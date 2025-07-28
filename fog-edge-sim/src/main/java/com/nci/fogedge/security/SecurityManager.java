package com.nci.fogedge.security;

import com.nci.fogedge.devices.*;
import com.nci.fogedge.core.SimulationConfig;
import com.nci.fogedge.core.SimulationResults;

import java.util.*;

/**
 * Manages security aspects of the simulation, including attack simulation,
 * device compromise detection, and security countermeasures.
 */
public class SecurityManager {
    private SimulationConfig config;
    private SimulationResults results;
    
    private boolean encryptionEnabled;
    private boolean intrusionDetectionEnabled;
    private boolean failCompromisedTasks;
    
    private Random random;
    private Set<Device> compromisedDevices;
    private List<SecurityEvent> securityEvents;
    
    private double detectionRate; // Probability of detecting a compromised device (0-1)
    private double falsePositiveRate; // Probability of false positive detection (0-1)
    
    /**
     * Constructor for SecurityManager
     * 
     * @param config Simulation configuration
     * @param results Simulation results
     */
    public SecurityManager(SimulationConfig config, SimulationResults results) {
        this.config = config;
        this.results = results;
        
        this.encryptionEnabled = config.isEncryptionEnabled();
        this.intrusionDetectionEnabled = config.isIntrusionDetectionEnabled();
        this.failCompromisedTasks = config.isFailCompromisedTasks();
        
        this.random = new Random(config.getRandomSeed());
        this.compromisedDevices = new HashSet<>();
        this.securityEvents = new ArrayList<>();
        
        this.detectionRate = config.getIntrusionDetectionRate();
        this.falsePositiveRate = config.getFalsePositiveRate();
    }
    
    /**
     * Simulates security attacks in the system
     * 
     * @param iotDevices List of IoT devices
     * @param edgeNodes List of edge nodes
     * @param fogNodes List of fog nodes
     * @param cloudDatacenters List of cloud datacenters
     * @param currentTime Current simulation time
     */
    public void simulateAttacks(List<IoTDevice> iotDevices, List<EdgeNode> edgeNodes,
                               List<FogNode> fogNodes, List<CloudDatacenter> cloudDatacenters,
                               double currentTime) {
        // Probability of attack per time step
        double attackProbability = config.getAttackProbability();
        
        // Check if an attack should occur
        if (random.nextDouble() < attackProbability) {
            // Select attack type
            AttackType attackType = selectRandomAttackType();
            
            // Select target device
            Device targetDevice = selectRandomDevice(iotDevices, edgeNodes, fogNodes, cloudDatacenters);
            
            if (targetDevice != null) {
                // Execute the attack
                executeAttack(attackType, targetDevice, currentTime);
            }
        }
        
        // Run intrusion detection if enabled
        if (intrusionDetectionEnabled) {
            runIntrusionDetection(iotDevices, edgeNodes, fogNodes, cloudDatacenters, currentTime);
        }
    }
    
    /**
     * Selects a random attack type based on configuration probabilities
     * 
     * @return Selected attack type
     */
    private AttackType selectRandomAttackType() {
        double rand = random.nextDouble();
        double cumulativeProbability = 0.0;
        
        // DDoS attack (30% probability)
        cumulativeProbability += 0.3;
        if (rand < cumulativeProbability) {
            return AttackType.DDOS;
        }
        
        // Data breach (20% probability)
        cumulativeProbability += 0.2;
        if (rand < cumulativeProbability) {
            return AttackType.DATA_BREACH;
        }
        
        // Man-in-the-middle (15% probability)
        cumulativeProbability += 0.15;
        if (rand < cumulativeProbability) {
            return AttackType.MAN_IN_THE_MIDDLE;
        }
        
        // Malware (25% probability)
        cumulativeProbability += 0.25;
        if (rand < cumulativeProbability) {
            return AttackType.MALWARE;
        }
        
        // Ransomware (10% probability)
        return AttackType.RANSOMWARE;
    }
    
    /**
     * Selects a random device from all devices in the simulation
     * 
     * @param iotDevices List of IoT devices
     * @param edgeNodes List of edge nodes
     * @param fogNodes List of fog nodes
     * @param cloudDatacenters List of cloud datacenters
     * @return Selected device
     */
    private Device selectRandomDevice(List<IoTDevice> iotDevices, List<EdgeNode> edgeNodes,
                                    List<FogNode> fogNodes, List<CloudDatacenter> cloudDatacenters) {
        List<Device> allDevices = new ArrayList<>();
        
        // Add all devices to the list with different weights based on vulnerability
        // IoT devices are most vulnerable
        for (IoTDevice device : iotDevices) {
            if (device.isActive()) {
                // Add IoT devices multiple times to increase their selection probability
                for (int i = 0; i < 5; i++) {
                    allDevices.add(device);
                }
            }
        }
        
        // Edge nodes are somewhat vulnerable
        for (EdgeNode node : edgeNodes) {
            if (node.isActive()) {
                // Add edge nodes multiple times but less than IoT devices
                for (int i = 0; i < 3; i++) {
                    allDevices.add(node);
                }
            }
        }
        
        // Fog nodes are less vulnerable
        for (FogNode node : fogNodes) {
            if (node.isActive()) {
                // Add fog nodes multiple times but less than edge nodes
                for (int i = 0; i < 2; i++) {
                    allDevices.add(node);
                }
            }
        }
        
        // Cloud datacenters are least vulnerable
        for (CloudDatacenter datacenter : cloudDatacenters) {
            if (datacenter.isActive()) {
                // Add cloud datacenters only once
                allDevices.add(datacenter);
            }
        }
        
        // Select a random device
        if (!allDevices.isEmpty()) {
            int index = random.nextInt(allDevices.size());
            return allDevices.get(index);
        }
        
        return null;
    }
    
    /**
     * Executes an attack on a target device
     * 
     * @param attackType Type of attack
     * @param targetDevice Target device
     * @param currentTime Current simulation time
     */
    private void executeAttack(AttackType attackType, Device targetDevice, double currentTime) {
        // Check if encryption is enabled and if it prevents the attack
        if (encryptionEnabled && random.nextDouble() < config.getEncryptionEffectiveness()) {
            // Attack prevented by encryption
            SecurityEvent event = new SecurityEvent(
                currentTime,
                attackType,
                targetDevice,
                SecurityEventType.ATTACK_PREVENTED,
                "Attack prevented by encryption"
            );
            
            securityEvents.add(event);
            results.incrementAttacksPrevented();
            
            return;
        }
        
        // Execute the attack based on its type
        switch (attackType) {
            case DDOS:
                executeDDoSAttack(targetDevice, currentTime);
                break;
                
            case DATA_BREACH:
                executeDataBreachAttack(targetDevice, currentTime);
                break;
                
            case MAN_IN_THE_MIDDLE:
                executeManInTheMiddleAttack(targetDevice, currentTime);
                break;
                
            case MALWARE:
                executeMalwareAttack(targetDevice, currentTime);
                break;
                
            case RANSOMWARE:
                executeRansomwareAttack(targetDevice, currentTime);
                break;
        }
        
        // Mark the device as compromised
        compromisedDevices.add(targetDevice);
        targetDevice.setCompromised(true);
        
        // Record the attack
        SecurityEvent event = new SecurityEvent(
            currentTime,
            attackType,
            targetDevice,
            SecurityEventType.ATTACK_SUCCESSFUL,
            "Device compromised by " + attackType
        );
        
        securityEvents.add(event);
        results.incrementAttacksSuccessful();
    }
    
    /**
     * Executes a DDoS attack on a target device
     * 
     * @param targetDevice Target device
     * @param currentTime Current simulation time
     */
    private void executeDDoSAttack(Device targetDevice, double currentTime) {
        // Simulate high resource utilization due to DDoS
        double utilization = 90.0 + (random.nextDouble() * 10.0); // 90-100%
        targetDevice.updateResourceUtilization(utilization);
    }
    
    /**
     * Executes a data breach attack on a target device
     * 
     * @param targetDevice Target device
     * @param currentTime Current simulation time
     */
    private void executeDataBreachAttack(Device targetDevice, double currentTime) {
        // Data breach doesn't affect device performance directly
        // It's more about data exfiltration
    }
    
    /**
     * Executes a man-in-the-middle attack on a target device
     * 
     * @param targetDevice Target device
     * @param currentTime Current simulation time
     */
    private void executeManInTheMiddleAttack(Device targetDevice, double currentTime) {
        // Man-in-the-middle attack doesn't affect device performance directly
        // It's more about intercepting communications
    }
    
    /**
     * Executes a malware attack on a target device
     * 
     * @param targetDevice Target device
     * @param currentTime Current simulation time
     */
    private void executeMalwareAttack(Device targetDevice, double currentTime) {
        // Simulate increased resource utilization due to malware
        double utilization = 70.0 + (random.nextDouble() * 20.0); // 70-90%
        targetDevice.updateResourceUtilization(utilization);
        
        // Simulate battery drain for IoT devices
        if (targetDevice instanceof IoTDevice) {
            double drainAmount = targetDevice.getBatteryCapacity() * 0.1; // Drain 10% of battery
            targetDevice.consumeEnergy(drainAmount);
        }
    }
    
    /**
     * Executes a ransomware attack on a target device
     * 
     * @param targetDevice Target device
     * @param currentTime Current simulation time
     */
    private void executeRansomwareAttack(Device targetDevice, double currentTime) {
        // Simulate complete resource utilization due to ransomware
        targetDevice.updateResourceUtilization(100.0);
    }
    
    /**
     * Runs intrusion detection on all devices
     * 
     * @param iotDevices List of IoT devices
     * @param edgeNodes List of edge nodes
     * @param fogNodes List of fog nodes
     * @param cloudDatacenters List of cloud datacenters
     * @param currentTime Current simulation time
     */
    private void runIntrusionDetection(List<IoTDevice> iotDevices, List<EdgeNode> edgeNodes,
                                     List<FogNode> fogNodes, List<CloudDatacenter> cloudDatacenters,
                                     double currentTime) {
        // Check all devices for intrusion
        checkDevicesForIntrusion(iotDevices, currentTime);
        checkDevicesForIntrusion(edgeNodes, currentTime);
        checkDevicesForIntrusion(fogNodes, currentTime);
        checkDevicesForIntrusion(cloudDatacenters, currentTime);
    }
    
    /**
     * Checks a list of devices for intrusion
     * 
     * @param devices List of devices
     * @param currentTime Current simulation time
     */
    private void checkDevicesForIntrusion(List<? extends Device> devices, double currentTime) {
        for (Device device : devices) {
            if (!device.isActive()) {
                continue;
            }
            
            // Check if the device is actually compromised
            if (device.isCompromised()) {
                // Probability of detecting the intrusion
                if (random.nextDouble() < detectionRate) {
                    // Intrusion detected
                    SecurityEvent event = new SecurityEvent(
                        currentTime,
                        null, // Unknown attack type
                        device,
                        SecurityEventType.INTRUSION_DETECTED,
                        "Intrusion detected on device"
                    );
                    
                    securityEvents.add(event);
                    results.incrementIntrusionsDetected();
                    
                    // Recover the device
                    recoverDevice(device, currentTime);
                }
            } else {
                // Device is not compromised, but might trigger a false positive
                if (random.nextDouble() < falsePositiveRate) {
                    // False positive
                    SecurityEvent event = new SecurityEvent(
                        currentTime,
                        null, // No attack
                        device,
                        SecurityEventType.FALSE_POSITIVE,
                        "False positive detection on device"
                    );
                    
                    securityEvents.add(event);
                    results.incrementFalsePositives();
                }
            }
        }
    }
    
    /**
     * Recovers a compromised device
     * 
     * @param device Compromised device
     * @param currentTime Current simulation time
     */
    private void recoverDevice(Device device, double currentTime) {
        // Remove from compromised devices
        compromisedDevices.remove(device);
        
        // Reset device state
        device.setCompromised(false);
        device.updateResourceUtilization(0.0);
        
        // Record recovery event
        SecurityEvent event = new SecurityEvent(
            currentTime,
            null, // Unknown attack type
            device,
            SecurityEventType.DEVICE_RECOVERED,
            "Device recovered from compromise"
        );
        
        securityEvents.add(event);
        results.incrementDevicesRecovered();
    }
    
    /**
     * Checks if encryption is enabled
     * 
     * @return True if encryption is enabled, false otherwise
     */
    public boolean isEncryptionEnabled() {
        return encryptionEnabled;
    }
    
    /**
     * Sets whether encryption is enabled
     * 
     * @param encryptionEnabled True if encryption is enabled, false otherwise
     */
    public void setEncryptionEnabled(boolean encryptionEnabled) {
        this.encryptionEnabled = encryptionEnabled;
    }
    
    /**
     * Checks if intrusion detection is enabled
     * 
     * @return True if intrusion detection is enabled, false otherwise
     */
    public boolean isIntrusionDetectionEnabled() {
        return intrusionDetectionEnabled;
    }
    
    /**
     * Sets whether intrusion detection is enabled
     * 
     * @param intrusionDetectionEnabled True if intrusion detection is enabled, false otherwise
     */
    public void setIntrusionDetectionEnabled(boolean intrusionDetectionEnabled) {
        this.intrusionDetectionEnabled = intrusionDetectionEnabled;
    }
    
    /**
     * Checks if compromised tasks should fail
     * 
     * @return True if compromised tasks should fail, false otherwise
     */
    public boolean shouldFailCompromisedTasks() {
        return failCompromisedTasks;
    }
    
    /**
     * Sets whether compromised tasks should fail
     * 
     * @param failCompromisedTasks True if compromised tasks should fail, false otherwise
     */
    public void setFailCompromisedTasks(boolean failCompromisedTasks) {
        this.failCompromisedTasks = failCompromisedTasks;
    }
    
    /**
     * Gets the set of compromised devices
     * 
     * @return Set of compromised devices
     */
    public Set<Device> getCompromisedDevices() {
        return new HashSet<>(compromisedDevices);
    }
    
    /**
     * Gets the list of security events
     * 
     * @return List of security events
     */
    public List<SecurityEvent> getSecurityEvents() {
        return new ArrayList<>(securityEvents);
    }
    
    /**
     * Gets the intrusion detection rate
     * 
     * @return Intrusion detection rate (0-1)
     */
    public double getDetectionRate() {
        return detectionRate;
    }
    
    /**
     * Sets the intrusion detection rate
     * 
     * @param detectionRate Intrusion detection rate (0-1)
     */
    public void setDetectionRate(double detectionRate) {
        this.detectionRate = Math.max(0, Math.min(1, detectionRate));
    }
    
    /**
     * Gets the false positive rate
     * 
     * @return False positive rate (0-1)
     */
    public double getFalsePositiveRate() {
        return falsePositiveRate;
    }
    
    /**
     * Sets the false positive rate
     * 
     * @param falsePositiveRate False positive rate (0-1)
     */
    public void setFalsePositiveRate(double falsePositiveRate) {
        this.falsePositiveRate = Math.max(0, Math.min(1, falsePositiveRate));
    }
}
