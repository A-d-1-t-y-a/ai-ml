package org.nci.fogedge.security;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.nci.fogedge.topology.EdgeNode;
import org.nci.fogedge.topology.FogNode;
import org.nci.fogedge.topology.IoTDevice;
import org.nci.fogedge.topology.NetworkTopology;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Class to simulate various attacks on the fog and edge computing topology
 * Based on the paper: "An Overview of Fog Computing and Edge Computing Security and Privacy Issues"
 * (Sensors 2021, 21, 8226, https://doi.org/10.3390/s21248226)
 */
public class AttackSimulator {
    private static final Logger logger = LogManager.getLogger(AttackSimulator.class);
    private static final Random random = new Random();
    
    private NetworkTopology topology;
    private List<AttackType> enabledAttackTypes;
    private Map<Integer, List<Attack>> activeAttacks; // Step -> List of active attacks
    private int totalAttacksLaunched;
    
    public AttackSimulator() {
        this.enabledAttackTypes = new ArrayList<>();
        this.activeAttacks = new HashMap<>();
        this.totalAttacksLaunched = 0;
    }
    
    /**
     * Initialize the attack simulator with the network topology and enabled attack types
     * @param topology The network topology to simulate attacks on
     * @param enabledAttackTypes List of enabled attack types
     */
    public void initialize(NetworkTopology topology, List<AttackType> enabledAttackTypes) {
        this.topology = topology;
        this.enabledAttackTypes = enabledAttackTypes;
        
        logger.info("Attack simulator initialized with {} attack types", enabledAttackTypes.size());
        for (AttackType type : enabledAttackTypes) {
            logger.debug("Enabled attack type: {}", type);
        }
    }
    
    /**
     * Simulate attacks for the current simulation step
     * @param step Current simulation step
     */
    public void simulateAttacks(int step) {
        if (topology == null || enabledAttackTypes.isEmpty()) {
            logger.warn("Attack simulator not properly initialized");
            return;
        }
        
        List<Attack> stepsActiveAttacks = new ArrayList<>();
        
        // Determine if we should launch a new attack in this step
        if (shouldLaunchAttack()) {
            // Select a random attack type from enabled types
            AttackType attackType = getRandomAttackType();
            
            // Select a target based on the attack type
            Object target = selectTargetForAttack(attackType);
            
            if (target != null) {
                // Create and launch the attack
                Attack attack = new Attack(attackType, target, step);
                stepsActiveAttacks.add(attack);
                totalAttacksLaunched++;
                
                // Apply the attack effects
                applyAttackEffects(attack);
                
                logger.warn("Launched new attack: {} on target {}", attackType, getTargetId(target));
            }
        }
        
        // Continue any ongoing attacks from previous steps
        for (Map.Entry<Integer, List<Attack>> entry : activeAttacks.entrySet()) {
            for (Attack attack : entry.getValue()) {
                // Check if the attack is still active
                if (attack.isActive(step)) {
                    stepsActiveAttacks.add(attack);
                    logger.debug("Continuing attack: {} on target {}", 
                            attack.getType(), getTargetId(attack.getTarget()));
                }
            }
        }
        
        // Store the active attacks for this step
        activeAttacks.put(step, stepsActiveAttacks);
        
        logger.info("Step {}: {} active attacks", step, stepsActiveAttacks.size());
    }
    
    /**
     * Determine if a new attack should be launched based on probability
     * @return true if a new attack should be launched
     */
    private boolean shouldLaunchAttack() {
        // Base probability of 10% for an attack in any step
        double attackProbability = 0.1;
        
        return random.nextDouble() < attackProbability;
    }
    
    /**
     * Select a random attack type from the enabled types
     * @return The selected attack type
     */
    private AttackType getRandomAttackType() {
        int index = random.nextInt(enabledAttackTypes.size());
        return enabledAttackTypes.get(index);
    }
    
    /**
     * Select a target for the given attack type
     * @param attackType The attack type
     * @return The selected target object
     */
    private Object selectTargetForAttack(AttackType attackType) {
        SecurityLayer targetLayer = attackType.getTargetLayer();
        
        switch (targetLayer) {
            case IOT:
                List<IoTDevice> devices = topology.getIoTDevices();
                if (!devices.isEmpty()) {
                    return devices.get(random.nextInt(devices.size()));
                }
                break;
                
            case EDGE:
                List<EdgeNode> edgeNodes = topology.getEdgeNodes();
                if (!edgeNodes.isEmpty()) {
                    return edgeNodes.get(random.nextInt(edgeNodes.size()));
                }
                break;
                
            case FOG:
                List<FogNode> fogNodes = topology.getFogNodes();
                if (!fogNodes.isEmpty()) {
                    return fogNodes.get(random.nextInt(fogNodes.size()));
                }
                break;
                
            case NETWORK:
                // For network attacks, we'll just select a random IoT device or edge node
                if (random.nextBoolean() && !topology.getIoTDevices().isEmpty()) {
                    return topology.getIoTDevices().get(random.nextInt(topology.getIoTDevices().size()));
                } else if (!topology.getEdgeNodes().isEmpty()) {
                    return topology.getEdgeNodes().get(random.nextInt(topology.getEdgeNodes().size()));
                }
                break;
        }
        
        logger.warn("Could not find suitable target for attack type: {}", attackType);
        return null;
    }
    
    /**
     * Apply the effects of an attack to the target
     * @param attack The attack to apply
     */
    private void applyAttackEffects(Attack attack) {
        Object target = attack.getTarget();
        AttackType type = attack.getType();
        
        // Apply different effects based on attack type and target
        if (target instanceof IoTDevice) {
            IoTDevice device = (IoTDevice) target;
            
            switch (type) {
                case IOT_PHYSICAL_TAMPERING:
                case IOT_MALWARE_INJECTION:
                    device.compromise();
                    break;
                    
                case IOT_BATTERY_DRAINING:
                    // Drain battery faster
                    device.consumeBattery(20.0);
                    break;
                    
                case NETWORK_EAVESDROPPING:
                case NETWORK_TRAFFIC_ANALYSIS:
                    // No visible effect, just logging
                    logger.debug("Network attack {} targeting device {}", type, device.getId());
                    break;
                    
                default:
                    logger.warn("Unsupported attack type {} for IoT device", type);
            }
        } else if (target instanceof EdgeNode) {
            EdgeNode edgeNode = (EdgeNode) target;
            
            switch (type) {
                case EDGE_DOS:
                case EDGE_MAN_IN_MIDDLE:
                case EDGE_AUTHENTICATION_BYPASS:
                    edgeNode.compromise();
                    break;
                    
                case NETWORK_EAVESDROPPING:
                case NETWORK_TRAFFIC_ANALYSIS:
                case NETWORK_ROUTING_ATTACK:
                    // No visible effect, just logging
                    logger.debug("Network attack {} targeting edge node {}", type, edgeNode.getId());
                    break;
                    
                default:
                    logger.warn("Unsupported attack type {} for edge node", type);
            }
        } else if (target instanceof FogNode) {
            FogNode fogNode = (FogNode) target;
            
            switch (type) {
                case FOG_DATA_THEFT:
                case FOG_PRIVILEGE_ESCALATION:
                case FOG_VM_ESCAPE:
                    fogNode.compromise();
                    break;
                    
                default:
                    logger.warn("Unsupported attack type {} for fog node", type);
            }
        }
    }
    
    /**
     * Get the ID of a target object
     * @param target The target object
     * @return The ID as a string
     */
    private String getTargetId(Object target) {
        if (target instanceof IoTDevice) {
            return ((IoTDevice) target).getId();
        } else if (target instanceof EdgeNode) {
            return ((EdgeNode) target).getId();
        } else if (target instanceof FogNode) {
            return ((FogNode) target).getId();
        }
        return "Unknown";
    }
    
    /**
     * Get the list of active attacks for the current step
     * @return List of active attacks
     */
    public List<Attack> getActiveAttacks() {
        // Return the most recent step's attacks
        if (activeAttacks.isEmpty()) {
            return new ArrayList<>();
        }
        
        int latestStep = activeAttacks.keySet().stream().max(Integer::compare).orElse(0);
        return activeAttacks.getOrDefault(latestStep, new ArrayList<>());
    }
    
    /**
     * Get the total number of attacks launched
     * @return Total attacks launched
     */
    public int getTotalAttacksLaunched() {
        return totalAttacksLaunched;
    }
    
    /**
     * Inner class representing an attack in the simulation
     */
    public static class Attack {
        private AttackType type;
        private Object target;
        private int startStep;
        private int duration;
        
        public Attack(AttackType type, Object target, int startStep) {
            this.type = type;
            this.target = target;
            this.startStep = startStep;
            
            // Random duration between 1-5 steps
            this.duration = 1 + new Random().nextInt(5);
        }
        
        /**
         * Check if the attack is still active in the given step
         * @param currentStep The current simulation step
         * @return true if the attack is active
         */
        public boolean isActive(int currentStep) {
            return currentStep >= startStep && currentStep < startStep + duration;
        }
        
        public AttackType getType() {
            return type;
        }
        
        public Object getTarget() {
            return target;
        }
        
        public int getStartStep() {
            return startStep;
        }
        
        public int getDuration() {
            return duration;
        }
        
        @Override
        public String toString() {
            String targetId = "Unknown";
            if (target instanceof IoTDevice) {
                targetId = ((IoTDevice) target).getId();
            } else if (target instanceof EdgeNode) {
                targetId = ((EdgeNode) target).getId();
            } else if (target instanceof FogNode) {
                targetId = ((FogNode) target).getId();
            }
            
            return type + " on " + targetId + " (Steps " + startStep + "-" + (startStep + duration - 1) + ")";
        }
    }
}
