package org.nci.fogedge.model;

import org.nci.fogedge.security.AttackType;
import org.nci.fogedge.security.SecurityLevel;
import org.nci.fogedge.topology.WirelessType;

import java.util.List;
import java.util.Random;

/**
 * Configuration class for the Fog and Edge Computing Security Simulation
 * Based on the paper: "An Overview of Fog Computing and Edge Computing Security and Privacy Issues"
 * (Sensors 2021, 21, 8226, https://doi.org/10.3390/s21248226)
 */
public class SimulationConfig {
    private int numIoTDevices;
    private int numEdgeNodes;
    private int numFogNodes;
    private int simulationSteps;
    private SecurityLevel securityLevel;
    private boolean securityEnabledAtIoT;
    private boolean securityEnabledAtEdge;
    private boolean securityEnabledAtFog;
    private boolean attackSimulationEnabled;
    private List<AttackType> attackTypes;
    private Random random;
    
    public SimulationConfig() {
        this.random = new Random();
    }
    
    public int getNumIoTDevices() {
        return numIoTDevices;
    }
    
    public void setNumIoTDevices(int numIoTDevices) {
        this.numIoTDevices = numIoTDevices;
    }
    
    public int getNumEdgeNodes() {
        return numEdgeNodes;
    }
    
    public void setNumEdgeNodes(int numEdgeNodes) {
        this.numEdgeNodes = numEdgeNodes;
    }
    
    public int getNumFogNodes() {
        return numFogNodes;
    }
    
    public void setNumFogNodes(int numFogNodes) {
        this.numFogNodes = numFogNodes;
    }
    
    public int getSimulationSteps() {
        return simulationSteps;
    }
    
    public void setSimulationSteps(int simulationSteps) {
        this.simulationSteps = simulationSteps;
    }
    
    public SecurityLevel getSecurityLevel() {
        return securityLevel;
    }
    
    public void setSecurityLevel(SecurityLevel securityLevel) {
        this.securityLevel = securityLevel;
    }
    
    public boolean isSecurityEnabledAtIoT() {
        return securityEnabledAtIoT;
    }
    
    public void setSecurityEnabledAtIoT(boolean securityEnabledAtIoT) {
        this.securityEnabledAtIoT = securityEnabledAtIoT;
    }
    
    public boolean isSecurityEnabledAtEdge() {
        return securityEnabledAtEdge;
    }
    
    public void setSecurityEnabledAtEdge(boolean securityEnabledAtEdge) {
        this.securityEnabledAtEdge = securityEnabledAtEdge;
    }
    
    public boolean isSecurityEnabledAtFog() {
        return securityEnabledAtFog;
    }
    
    public void setSecurityEnabledAtFog(boolean securityEnabledAtFog) {
        this.securityEnabledAtFog = securityEnabledAtFog;
    }
    
    public boolean isAttackSimulationEnabled() {
        return attackSimulationEnabled;
    }
    
    public void setAttackSimulationEnabled(boolean attackSimulationEnabled) {
        this.attackSimulationEnabled = attackSimulationEnabled;
    }
    
    public List<AttackType> getAttackTypes() {
        return attackTypes;
    }
    
    public void setAttackTypes(List<AttackType> attackTypes) {
        this.attackTypes = attackTypes;
    }
    
    /**
     * Returns a random wireless type for IoT devices
     * Based on the paper's discussion of different wireless technologies
     */
    public WirelessType getRandomWirelessType() {
        WirelessType[] types = WirelessType.values();
        return types[random.nextInt(types.length)];
    }
    
    @Override
    public String toString() {
        return "SimulationConfig{" +
                "numIoTDevices=" + numIoTDevices +
                ", numEdgeNodes=" + numEdgeNodes +
                ", numFogNodes=" + numFogNodes +
                ", simulationSteps=" + simulationSteps +
                ", securityLevel=" + securityLevel +
                ", securityEnabledAtIoT=" + securityEnabledAtIoT +
                ", securityEnabledAtEdge=" + securityEnabledAtEdge +
                ", securityEnabledAtFog=" + securityEnabledAtFog +
                ", attackSimulationEnabled=" + attackSimulationEnabled +
                ", attackTypes=" + attackTypes +
                '}';
    }
}
