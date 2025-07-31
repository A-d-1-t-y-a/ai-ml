package com.fog.eedto.simulation;

/**
 * Stores and provides access to the results of an EEDTO simulation.
 * This class contains various metrics collected during the simulation.
 */
public class SimulationResults {
    private final int totalTasksGenerated;
    private final int totalTasksCompleted;
    private final int totalTasksRejected;
    private final double totalEnergyConsumed;
    private final double totalResponseTime;
    private final double totalExecutionCost;
    private final int localExecutions;
    private final int edgeOffloads;
    private final int cloudOffloads;
    private final int failedOffloads;
    private final int blockchainSize;
    
    /**
     * Constructor for the SimulationResults class
     * 
     * @param totalTasksGenerated Total number of tasks generated
     * @param totalTasksCompleted Total number of tasks completed
     * @param totalTasksRejected Total number of tasks rejected
     * @param totalEnergyConsumed Total energy consumed in Joules
     * @param totalResponseTime Total response time in seconds
     * @param totalExecutionCost Total execution cost in monetary units
     * @param localExecutions Number of tasks executed locally
     * @param edgeOffloads Number of tasks offloaded to edge servers
     * @param cloudOffloads Number of tasks offloaded to cloud servers
     * @param failedOffloads Number of tasks that failed to be offloaded
     * @param blockchainSize Number of blocks in the blockchain
     */
    public SimulationResults(int totalTasksGenerated, int totalTasksCompleted, int totalTasksRejected,
                            double totalEnergyConsumed, double totalResponseTime, double totalExecutionCost,
                            int localExecutions, int edgeOffloads, int cloudOffloads, int failedOffloads,
                            int blockchainSize) {
        this.totalTasksGenerated = totalTasksGenerated;
        this.totalTasksCompleted = totalTasksCompleted;
        this.totalTasksRejected = totalTasksRejected;
        this.totalEnergyConsumed = totalEnergyConsumed;
        this.totalResponseTime = totalResponseTime;
        this.totalExecutionCost = totalExecutionCost;
        this.localExecutions = localExecutions;
        this.edgeOffloads = edgeOffloads;
        this.cloudOffloads = cloudOffloads;
        this.failedOffloads = failedOffloads;
        this.blockchainSize = blockchainSize;
    }
    
    // Getters
    public int getTotalTasksGenerated() {
        return totalTasksGenerated;
    }
    
    public int getTotalTasksCompleted() {
        return totalTasksCompleted;
    }
    
    public int getTotalTasksRejected() {
        return totalTasksRejected;
    }
    
    public double getTotalEnergyConsumed() {
        return totalEnergyConsumed;
    }
    
    public double getTotalResponseTime() {
        return totalResponseTime;
    }
    
    public double getTotalExecutionCost() {
        return totalExecutionCost;
    }
    
    public int getLocalExecutions() {
        return localExecutions;
    }
    
    public int getEdgeOffloads() {
        return edgeOffloads;
    }
    
    public int getCloudOffloads() {
        return cloudOffloads;
    }
    
    public int getFailedOffloads() {
        return failedOffloads;
    }
    
    public int getBlockchainSize() {
        return blockchainSize;
    }
    
    // Derived metrics
    public double getTaskCompletionRate() {
        return totalTasksGenerated > 0 ? (double) totalTasksCompleted / totalTasksGenerated * 100 : 0;
    }
    
    public double getTaskRejectionRate() {
        return totalTasksGenerated > 0 ? (double) totalTasksRejected / totalTasksGenerated * 100 : 0;
    }
    
    public double getAverageEnergyPerTask() {
        return totalTasksCompleted > 0 ? totalEnergyConsumed / totalTasksCompleted : 0;
    }
    
    public double getAverageResponseTime() {
        return totalTasksCompleted > 0 ? totalResponseTime / totalTasksCompleted : 0;
    }
    
    public double getAverageExecutionCost() {
        return totalTasksCompleted > 0 ? totalExecutionCost / totalTasksCompleted : 0;
    }
    
    public double getLocalExecutionPercentage() {
        return totalTasksCompleted > 0 ? (double) localExecutions / totalTasksCompleted * 100 : 0;
    }
    
    public double getEdgeOffloadPercentage() {
        return totalTasksCompleted > 0 ? (double) edgeOffloads / totalTasksCompleted * 100 : 0;
    }
    
    public double getCloudOffloadPercentage() {
        return totalTasksCompleted > 0 ? (double) cloudOffloads / totalTasksCompleted * 100 : 0;
    }
    
    public double getFailedOffloadPercentage() {
        return totalTasksGenerated > 0 ? (double) failedOffloads / totalTasksGenerated * 100 : 0;
    }
    
    @Override
    public String toString() {
        return "SimulationResults{\n" +
                "  totalTasksGenerated=" + totalTasksGenerated + "\n" +
                "  totalTasksCompleted=" + totalTasksCompleted + " (" + String.format("%.2f", getTaskCompletionRate()) + "%)\n" +
                "  totalTasksRejected=" + totalTasksRejected + " (" + String.format("%.2f", getTaskRejectionRate()) + "%)\n" +
                "  averageEnergyPerTask=" + String.format("%.2f", getAverageEnergyPerTask()) + " J\n" +
                "  averageResponseTime=" + String.format("%.2f", getAverageResponseTime()) + " s\n" +
                "  averageExecutionCost=$" + String.format("%.6f", getAverageExecutionCost()) + "\n" +
                "  localExecutions=" + localExecutions + " (" + String.format("%.2f", getLocalExecutionPercentage()) + "%)\n" +
                "  edgeOffloads=" + edgeOffloads + " (" + String.format("%.2f", getEdgeOffloadPercentage()) + "%)\n" +
                "  cloudOffloads=" + cloudOffloads + " (" + String.format("%.2f", getCloudOffloadPercentage()) + "%)\n" +
                "  failedOffloads=" + failedOffloads + " (" + String.format("%.2f", getFailedOffloadPercentage()) + "%)\n" +
                "  blockchainSize=" + blockchainSize + " blocks\n" +
                '}';
    }
}
