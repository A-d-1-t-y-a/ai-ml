package com.nci.fogedge.security;

import com.nci.fogedge.devices.*;
import com.nci.fogedge.model.SimulationConfig;
import com.nci.fogedge.model.SimulationResults;
import com.nci.fogedge.tasks.Task;

import java.util.*;

/**
 * Manages security aspects of the simulation, including attack simulation and countermeasures.
 * This class is responsible for simulating various security attacks and defense mechanisms.
 */
public class SecurityManager {
    private SimulationConfig config;
    private SimulationResults results;
    private Map<String, AttackSimulation> activeAttacks;
    private Map<String, SecurityMeasure> activeMeasures;
    private Map<String, Boolean> compromisedDevices;
    private Random random;
    
    /**
     * Constructor for SecurityManager
     * 
     * @param config Simulation configuration
     * @param results Simulation results collector
     */
    public SecurityManager(SimulationConfig config, SimulationResults results) {
        this.config = config;
        this.results = results;
        this.activeAttacks = new HashMap<>();
        this.activeMeasures = new HashMap<>();
        this.compromisedDevices = new HashMap<>();
        this.random = new Random();
    }
    
    /**
     * Initializes the SecurityManager
     */
    public void initialize() {
        activeAttacks.clear();
        activeMeasures.clear();
        compromisedDevices.clear();
    }
    
    /**
     * Simulates security attacks based on configuration
     * 
     * @param devices Map of all devices indexed by ID
     * @param currentTick Current simulation tick
     */
    public void simulateAttacks(Map<String, Device> devices, int currentTick) {
        // Get attack probability from configuration
        double attackProbability = config.getAttackProbability();
        
        // Check if an attack should be generated this tick
        if (random.nextDouble() < attackProbability) {
            // Select a random device to attack
            List<String> deviceIds = new ArrayList<>(devices.keySet());
            if (deviceIds.isEmpty()) {
                return;
            }
            
            String targetDeviceId = deviceIds.get(random.nextInt(deviceIds.size()));
            Device targetDevice = devices.get(targetDeviceId);
            
            // Select a random attack type
            AttackType attackType = selectRandomAttackType();
            
            // Create and launch the attack
            launchAttack(targetDevice, attackType, currentTick);
        }
        
        // Update existing attacks
        updateActiveAttacks(devices, currentTick);
    }
    
    /**
     * Selects a random attack type
     * 
     * @return Random attack type
     */
    private AttackType selectRandomAttackType() {
        AttackType[] attackTypes = AttackType.values();
        return attackTypes[random.nextInt(attackTypes.length)];
    }
    
    /**
     * Launches an attack on a target device
     * 
     * @param targetDevice Target device
     * @param attackType Type of attack
     * @param currentTick Current simulation tick
     */
    private void launchAttack(Device targetDevice, AttackType attackType, int currentTick) {
        // Create attack ID
        String attackId = "attack_" + currentTick + "_" + targetDevice.getId();
        
        // Create attack simulation
        AttackSimulation attack = new AttackSimulation(
            attackId,
            targetDevice.getId(),
            attackType,
            currentTick,
            currentTick + calculateAttackDuration(attackType),
            calculateAttackSeverity(attackType)
        );
        
        // Add to active attacks
        activeAttacks.put(attackId, attack);
        
        // Update statistics
        results.incrementTotalAttacks();
        results.incrementAttacksByType(attackType);
        
        // Mark the device as potentially compromised
        // Whether it's actually compromised depends on the device's security level
        // and will be determined in the updateActiveAttacks method
        compromisedDevices.put(targetDevice.getId(), false);
    }
    
    /**
     * Calculates the duration of an attack based on its type
     * 
     * @param attackType Type of attack
     * @return Attack duration in simulation ticks
     */
    private int calculateAttackDuration(AttackType attackType) {
        switch (attackType) {
            case DDOS:
                return 20 + random.nextInt(30); // 20-50 ticks
                
            case DATA_THEFT:
                return 5 + random.nextInt(10); // 5-15 ticks
                
            case EAVESDROPPING:
                return 30 + random.nextInt(70); // 30-100 ticks
                
            case MAN_IN_THE_MIDDLE:
                return 15 + random.nextInt(25); // 15-40 ticks
                
            case MALWARE:
                return 50 + random.nextInt(100); // 50-150 ticks
                
            case PHYSICAL_TAMPERING:
                return 10 + random.nextInt(20); // 10-30 ticks
                
            default:
                return 20; // Default: 20 ticks
        }
    }
    
    /**
     * Calculates the severity of an attack based on its type
     * 
     * @param attackType Type of attack
     * @return Attack severity (0-1)
     */
    private double calculateAttackSeverity(AttackType attackType) {
        switch (attackType) {
            case DDOS:
                return 0.7 + random.nextDouble() * 0.3; // 0.7-1.0
                
            case DATA_THEFT:
                return 0.5 + random.nextDouble() * 0.3; // 0.5-0.8
                
            case EAVESDROPPING:
                return 0.3 + random.nextDouble() * 0.3; // 0.3-0.6
                
            case MAN_IN_THE_MIDDLE:
                return 0.6 + random.nextDouble() * 0.3; // 0.6-0.9
                
            case MALWARE:
                return 0.8 + random.nextDouble() * 0.2; // 0.8-1.0
                
            case PHYSICAL_TAMPERING:
                return 0.9 + random.nextDouble() * 0.1; // 0.9-1.0
                
            default:
                return 0.5; // Default: 0.5
        }
    }
    
    /**
     * Updates active attacks
     * 
     * @param devices Map of all devices indexed by ID
     * @param currentTick Current simulation tick
     */
    private void updateActiveAttacks(Map<String, Device> devices, int currentTick) {
        List<String> completedAttacks = new ArrayList<>();
        
        for (AttackSimulation attack : activeAttacks.values()) {
            // Check if the attack has completed
            if (currentTick >= attack.getEndTick()) {
                completedAttacks.add(attack.getId());
                continue;
            }
            
            // Get the target device
            String targetDeviceId = attack.getTargetDeviceId();
            Device targetDevice = devices.get(targetDeviceId);
            
            if (targetDevice == null) {
                // Target device no longer exists
                completedAttacks.add(attack.getId());
                continue;
            }
            
            // Check if the attack is detected
            boolean detected = detectAttack(targetDevice, attack);
            
            if (detected) {
                // Attack detected, apply countermeasures
                applyCountermeasures(targetDevice, attack, currentTick);
                
                // Update statistics
                results.incrementDetectedAttacks();
                
                // Mark the attack as completed
                completedAttacks.add(attack.getId());
            } else {
                // Attack not detected, apply attack effects
                applyAttackEffects(targetDevice, attack);
                
                // Mark the device as compromised
                compromisedDevices.put(targetDeviceId, true);
            }
        }
        
        // Remove completed attacks
        for (String attackId : completedAttacks) {
            activeAttacks.remove(attackId);
        }
    }
    
    /**
     * Detects if an attack is detected by a device
     * 
     * @param device Target device
     * @param attack Attack simulation
     * @return True if the attack is detected, false otherwise
     */
    private boolean detectAttack(Device device, AttackSimulation attack) {
        // Get the device's security level
        double securityLevel = 0.5; // Default security level
        
        if (device instanceof EdgeNode) {
            securityLevel = ((EdgeNode) device).getSecurityLevel();
        } else if (device instanceof FogNode) {
            securityLevel = ((FogNode) device).getSecurityLevel();
        } else if (device instanceof CloudDatacenter) {
            securityLevel = ((CloudDatacenter) device).getSecurityLevel();
        }
        
        // Calculate detection probability based on security level and attack severity
        double detectionProbability = securityLevel * (1.0 - attack.getSeverity() * 0.3);
        
        // Check if the attack is detected
        return random.nextDouble() < detectionProbability;
    }
    
    /**
     * Applies countermeasures to an attack
     * 
     * @param device Target device
     * @param attack Attack simulation
     * @param currentTick Current simulation tick
     */
    private void applyCountermeasures(Device device, AttackSimulation attack, int currentTick) {
        // Create countermeasure ID
        String measureId = "measure_" + currentTick + "_" + device.getId();
        
        // Select appropriate countermeasure based on attack type
        CountermeasureType measureType = selectCountermeasure(attack.getType());
        
        // Create security measure
        SecurityMeasure measure = new SecurityMeasure(
            measureId,
            device.getId(),
            attack.getId(),
            measureType,
            currentTick,
            currentTick + calculateMeasureDuration(measureType),
            calculateMeasureEffectiveness(measureType)
        );
        
        // Add to active measures
        activeMeasures.put(measureId, measure);
        
        // Apply immediate effects of the countermeasure
        applyCountermeasureEffects(device, measure);
        
        // Update statistics
        results.incrementCountermeasuresByType(measureType);
    }
    
    /**
     * Selects an appropriate countermeasure for an attack type
     * 
     * @param attackType Type of attack
     * @return Appropriate countermeasure type
     */
    private CountermeasureType selectCountermeasure(AttackType attackType) {
        switch (attackType) {
            case DDOS:
                return CountermeasureType.TRAFFIC_FILTERING;
                
            case DATA_THEFT:
                return CountermeasureType.ENCRYPTION;
                
            case EAVESDROPPING:
                return CountermeasureType.SECURE_COMMUNICATION;
                
            case MAN_IN_THE_MIDDLE:
                return CountermeasureType.AUTHENTICATION;
                
            case MALWARE:
                return CountermeasureType.MALWARE_SCANNING;
                
            case PHYSICAL_TAMPERING:
                return CountermeasureType.PHYSICAL_SECURITY;
                
            default:
                return CountermeasureType.INTRUSION_DETECTION;
        }
    }
    
    /**
     * Calculates the duration of a countermeasure based on its type
     * 
     * @param measureType Type of countermeasure
     * @return Countermeasure duration in simulation ticks
     */
    private int calculateMeasureDuration(CountermeasureType measureType) {
        switch (measureType) {
            case TRAFFIC_FILTERING:
                return 30 + random.nextInt(30); // 30-60 ticks
                
            case ENCRYPTION:
                return 100 + random.nextInt(100); // 100-200 ticks
                
            case SECURE_COMMUNICATION:
                return 50 + random.nextInt(50); // 50-100 ticks
                
            case AUTHENTICATION:
                return 40 + random.nextInt(40); // 40-80 ticks
                
            case MALWARE_SCANNING:
                return 20 + random.nextInt(20); // 20-40 ticks
                
            case PHYSICAL_SECURITY:
                return 200 + random.nextInt(100); // 200-300 ticks
                
            case INTRUSION_DETECTION:
                return 50 + random.nextInt(50); // 50-100 ticks
                
            default:
                return 50; // Default: 50 ticks
        }
    }
    
    /**
     * Calculates the effectiveness of a countermeasure based on its type
     * 
     * @param measureType Type of countermeasure
     * @return Countermeasure effectiveness (0-1)
     */
    private double calculateMeasureEffectiveness(CountermeasureType measureType) {
        switch (measureType) {
            case TRAFFIC_FILTERING:
                return 0.7 + random.nextDouble() * 0.2; // 0.7-0.9
                
            case ENCRYPTION:
                return 0.8 + random.nextDouble() * 0.2; // 0.8-1.0
                
            case SECURE_COMMUNICATION:
                return 0.7 + random.nextDouble() * 0.2; // 0.7-0.9
                
            case AUTHENTICATION:
                return 0.8 + random.nextDouble() * 0.1; // 0.8-0.9
                
            case MALWARE_SCANNING:
                return 0.6 + random.nextDouble() * 0.3; // 0.6-0.9
                
            case PHYSICAL_SECURITY:
                return 0.9 + random.nextDouble() * 0.1; // 0.9-1.0
                
            case INTRUSION_DETECTION:
                return 0.7 + random.nextDouble() * 0.2; // 0.7-0.9
                
            default:
                return 0.7; // Default: 0.7
        }
    }
    
    /**
     * Applies the effects of an attack to a device
     * 
     * @param device Target device
     * @param attack Attack simulation
     */
    private void applyAttackEffects(Device device, AttackSimulation attack) {
        // Apply effects based on attack type and severity
        switch (attack.getType()) {
            case DDOS:
                // Increase resource utilization
                double utilizationIncrease = attack.getSeverity() * 50; // 0-50% increase
                device.updateResourceUtilization(Math.min(100, device.getResourceUtilization() + utilizationIncrease));
                break;
                
            case DATA_THEFT:
                // No immediate effect on device performance, but data is compromised
                break;
                
            case EAVESDROPPING:
                // No immediate effect on device performance, but communications are compromised
                break;
                
            case MAN_IN_THE_MIDDLE:
                // No immediate effect on device performance, but communications are compromised
                break;
                
            case MALWARE:
                // Decrease device performance
                double utilizationDecrease = attack.getSeverity() * 30; // 0-30% decrease
                device.updateResourceUtilization(Math.min(100, device.getResourceUtilization() + utilizationDecrease));
                // Consume energy
                device.consumeEnergy(attack.getSeverity() * 0.2);
                break;
                
            case PHYSICAL_TAMPERING:
                // Device may be disabled
                if (attack.getSeverity() > 0.8) {
                    device.setActive(false);
                }
                break;
        }
    }
    
    /**
     * Applies the effects of a countermeasure to a device
     * 
     * @param device Target device
     * @param measure Security measure
     */
    private void applyCountermeasureEffects(Device device, SecurityMeasure measure) {
        // Apply effects based on countermeasure type
        switch (measure.getType()) {
            case TRAFFIC_FILTERING:
                // Increase resource utilization due to filtering overhead
                double utilizationIncrease = measure.getEffectiveness() * 10; // 0-10% increase
                device.updateResourceUtilization(Math.min(100, device.getResourceUtilization() + utilizationIncrease));
                break;
                
            case ENCRYPTION:
                // Increase resource utilization due to encryption overhead
                utilizationIncrease = measure.getEffectiveness() * 15; // 0-15% increase
                device.updateResourceUtilization(Math.min(100, device.getResourceUtilization() + utilizationIncrease));
                break;
                
            case SECURE_COMMUNICATION:
                // Increase resource utilization due to secure communication overhead
                utilizationIncrease = measure.getEffectiveness() * 12; // 0-12% increase
                device.updateResourceUtilization(Math.min(100, device.getResourceUtilization() + utilizationIncrease));
                break;
                
            case AUTHENTICATION:
                // Increase resource utilization due to authentication overhead
                utilizationIncrease = measure.getEffectiveness() * 8; // 0-8% increase
                device.updateResourceUtilization(Math.min(100, device.getResourceUtilization() + utilizationIncrease));
                break;
                
            case MALWARE_SCANNING:
                // Increase resource utilization due to scanning overhead
                utilizationIncrease = measure.getEffectiveness() * 20; // 0-20% increase
                device.updateResourceUtilization(Math.min(100, device.getResourceUtilization() + utilizationIncrease));
                break;
                
            case PHYSICAL_SECURITY:
                // No immediate effect on device performance
                break;
                
            case INTRUSION_DETECTION:
                // Increase resource utilization due to monitoring overhead
                utilizationIncrease = measure.getEffectiveness() * 10; // 0-10% increase
                device.updateResourceUtilization(Math.min(100, device.getResourceUtilization() + utilizationIncrease));
                break;
        }
        
        // Mark the device as no longer compromised
        compromisedDevices.put(device.getId(), false);
    }
    
    /**
     * Updates active security measures
     * 
     * @param devices Map of all devices indexed by ID
     * @param currentTick Current simulation tick
     */
    public void updateSecurityMeasures(Map<String, Device> devices, int currentTick) {
        List<String> completedMeasures = new ArrayList<>();
        
        for (SecurityMeasure measure : activeMeasures.values()) {
            // Check if the measure has completed
            if (currentTick >= measure.getEndTick()) {
                completedMeasures.add(measure.getId());
                continue;
            }
            
            // Get the target device
            String targetDeviceId = measure.getTargetDeviceId();
            Device targetDevice = devices.get(targetDeviceId);
            
            if (targetDevice == null) {
                // Target device no longer exists
                completedMeasures.add(measure.getId());
                continue;
            }
            
            // Continue applying the measure effects
            // This is a simplified version; in a real simulation, the effects would be more complex
            // and would depend on the specific measure type and the device state
        }
        
        // Remove completed measures
        for (String measureId : completedMeasures) {
            activeMeasures.remove(measureId);
        }
    }
    
    /**
     * Checks if a device is compromised
     * 
     * @param deviceId Device ID
     * @return True if the device is compromised, false otherwise
     */
    public boolean isDeviceCompromised(String deviceId) {
        return compromisedDevices.getOrDefault(deviceId, false);
    }
    
    /**
     * Checks if a task is affected by security issues
     * 
     * @param task Task to check
     * @return True if the task is affected, false otherwise
     */
    public boolean isTaskAffectedBySecurityIssues(Task task) {
        // Check if the source device is compromised
        String sourceDeviceId = task.getSourceDeviceId();
        if (isDeviceCompromised(sourceDeviceId)) {
            return true;
        }
        
        // Check if the executor device is compromised
        String executorDeviceId = task.getExecutorDeviceId();
        if (executorDeviceId != null && isDeviceCompromised(executorDeviceId)) {
            return true;
        }
        
        return false;
    }
    
    /**
     * Gets all active attacks
     * 
     * @return Map of active attacks indexed by ID
     */
    public Map<String, AttackSimulation> getActiveAttacks() {
        return activeAttacks;
    }
    
    /**
     * Gets all active security measures
     * 
     * @return Map of active security measures indexed by ID
     */
    public Map<String, SecurityMeasure> getActiveMeasures() {
        return activeMeasures;
    }
    
    /**
     * Gets the map of compromised devices
     * 
     * @return Map of device IDs to compromise status
     */
    public Map<String, Boolean> getCompromisedDevices() {
        return compromisedDevices;
    }
}
