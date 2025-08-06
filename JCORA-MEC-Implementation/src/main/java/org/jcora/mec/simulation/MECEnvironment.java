package org.jcora.mec.simulation;

import org.jcora.mec.drl.DRLAgent;
import org.jcora.mec.model.EdgeServer;
import org.jcora.mec.model.IoTDevice;
import org.jcora.mec.model.Task;
import org.nd4j.linalg.api.ndarray.INDArray;

// Use simple System.out logging instead of SLF4J for now
// import org.slf4j.Logger;
// import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Simulation environment for the Mobile Edge Computing (MEC) system.
 * This class coordinates the interaction between IoT devices, edge servers, and the DRL agent.
 */
public class MECEnvironment {
    // Use simple System.out logging instead of SLF4J for now
    // private static final Logger logger = LoggerFactory.getLogger(MECEnvironment.class);
    
    private final List<IoTDevice> devices;
    private final List<EdgeServer> servers;
    private final DRLAgent agent;
    private final Random random;
    
    private double currentTime;
    private int totalTasks;
    private int completedTasks;
    private int failedTasks;
    private double totalEnergyConsumed;
    private double totalResponseTime;
    private double totalDeadlineMissRate;
    
    // Simulation parameters
    private final double simulationDuration;
    private final double timeStep;
    private final double taskGenerationProbability;
    
    // Reward calculation parameters
    private final double maxEnergyForNormalization;
    private final double maxTimeForNormalization;
    private final double energyWeight;
    private final double timeWeight;
    private final double deadlineBonusReward;
    private final double deadlinePenaltyReward;
    
    // Performance metrics
    private final List<Double> energyConsumptionHistory;
    private final List<Double> responseTimeHistory;
    private final List<Double> deadlineMissRateHistory;
    private final List<Double> taskCompletionRateHistory;
    
    /**
     * Constructor for creating a new MEC environment.
     * 
     * @param devices List of IoT devices
     * @param servers List of edge servers
     * @param agent DRL agent for decision making
     * @param simulationDuration Total duration of the simulation in seconds
     * @param timeStep Time step for the simulation in seconds
     * @param taskGenerationProbability Probability of generating a new task at each time step
     */
    public MECEnvironment(List<IoTDevice> devices, List<EdgeServer> servers, DRLAgent agent,
                         double simulationDuration, double timeStep, double taskGenerationProbability,
                         double maxEnergyForNormalization, double maxTimeForNormalization,
                         double energyWeight, double timeWeight,
                         double deadlineBonusReward, double deadlinePenaltyReward) {
        this.devices = devices;
        this.servers = servers;
        this.agent = agent;
        this.simulationDuration = simulationDuration;
        this.timeStep = timeStep;
        this.taskGenerationProbability = taskGenerationProbability;
        
        // Initialize reward calculation parameters
        this.maxEnergyForNormalization = maxEnergyForNormalization > 0 ? maxEnergyForNormalization : 100.0;
        this.maxTimeForNormalization = maxTimeForNormalization > 0 ? maxTimeForNormalization : 10.0;
        this.energyWeight = energyWeight;
        this.timeWeight = timeWeight;
        this.deadlineBonusReward = deadlineBonusReward;
        this.deadlinePenaltyReward = deadlinePenaltyReward;
        
        this.random = new Random();
        this.currentTime = 0.0;
        this.totalTasks = 0;
        this.completedTasks = 0;
        this.failedTasks = 0;
        this.totalEnergyConsumed = 0.0;
        this.totalResponseTime = 0.0;
        this.totalDeadlineMissRate = 0.0;
        
        this.energyConsumptionHistory = new ArrayList<>();
        this.responseTimeHistory = new ArrayList<>();
        this.deadlineMissRateHistory = new ArrayList<>();
        this.taskCompletionRateHistory = new ArrayList<>();
    }
    
    /**
     * Run the simulation.
     */
    public void runSimulation() {
        System.out.println("Starting simulation with " + devices.size() + " devices and " + servers.size() + " servers");
        
        int step = 0;
        while (currentTime < simulationDuration) {
            // System.out.println("Simulation step " + step + " at time " + currentTime);
            
            // Generate new tasks
            generateTasks();
            
            // Process tasks using DRL agent
            processTasks();
            
            // Update system state
            updateSystemState();
            
            // Record metrics
            recordMetrics();
            
            // Advance time
            currentTime += timeStep;
            step++;
        }
        
        System.out.println("Simulation completed after " + step + " steps");
        System.out.println("Total tasks: " + totalTasks + ", Completed: " + completedTasks + ", Failed: " + failedTasks);
        System.out.println("Average energy consumption: " + (totalEnergyConsumed / totalTasks) + " J");
        System.out.println("Average response time: " + (totalResponseTime / completedTasks) + " s");
        System.out.println("Average deadline miss rate: " + ((totalDeadlineMissRate / totalTasks) * 100) + "%");
    }
    
    /**
     * Generate new tasks for IoT devices.
     */
    private void generateTasks() {
        for (IoTDevice device : devices) {
            // Generate a task with the specified probability
            if (random.nextDouble() < taskGenerationProbability) {
                // Create a new task with random parameters
                int taskId = totalTasks++;
                double inputDataSize = 1.0 + random.nextDouble() * 9.0; // 1-10 MB
                double outputDataSize = 0.1 + random.nextDouble() * 0.9; // 0.1-1 MB
                long computationalRequirement = 100 + random.nextInt(900); // 100-1000 MI
                double deadline = 1.0 + random.nextDouble() * 9.0; // 1-10 seconds
                
                Task task = new Task(taskId, inputDataSize, outputDataSize, computationalRequirement, 
                                    deadline, currentTime);
                
                // Add the task to the device
                device.addTask(task);
                
                // System.out.println("Generated task " + taskId + " for device " + device.getId());
            }
        }
    }
    
    /**
     * Process tasks using the DRL agent for decision making.
     */
    private void processTasks() {
        for (IoTDevice device : devices) {
            List<Task> taskQueue = device.getTaskQueue();
            Iterator<Task> iterator = taskQueue.iterator();
            
            while (iterator.hasNext()) {
                Task task = iterator.next();
                
                // Skip tasks that are already being processed or completed
                if (task.getStatus() != Task.TaskStatus.CREATED) {
                    continue;
                }
                
                // Check if DRL agent is available
                if (agent != null) {
                    // Get the current state
                    INDArray state = agent.getState(device, servers, task);
                    
                    // Select an action using the DRL agent
                    int action = agent.selectAction(state);
                    
                    // Execute the action
                    double reward = executeAction(device, task, action);
                    
                    // Get the next state
                    INDArray nextState = agent.getState(device, servers, task);
                    
                    // Store the experience in the agent's replay memory
                    boolean done = task.getStatus() == Task.TaskStatus.COMPLETED || 
                                  task.getStatus() == Task.TaskStatus.FAILED;
                    agent.remember(state, action, reward, nextState, done);
                } else {
                    // DRL agent is disabled, use a simple round-robin approach
                    System.out.println("DRL agent disabled, using round-robin for task: " + task.getId());
                    int serverIndex = (int)(Math.random() * servers.size());
                    EdgeServer server = servers.get(serverIndex);
                    executeAction(device, task, serverIndex);
                }
                
                // Remove the task from the queue if it's being processed
                if (task.getStatus() != Task.TaskStatus.CREATED) {
                    iterator.remove();
                }
            }
        }
        
        // Train the agent
        if (agent != null) {
            agent.train((int)(currentTime / timeStep));
        }
    }
    
    /**
     * Execute an action selected by the DRL agent.
     * 
     * @param device IoT device
     * @param task Task to be processed
     * @param action Selected action
     * @return Reward for the action
     */
    private double executeAction(IoTDevice device, Task task, int action) {
        // Action space:
        // 0: Process locally
        // 1 to N: Offload to server i-1 (where N is the number of servers)
        
        double reward = 0.0;
        
        if (action == 0) {
            // Process the task locally
            boolean success = device.processTaskLocally(task, currentTime);
            
            if (success) {
                // Calculate reward based on energy consumption and response time
                double energyConsumed = device.calculateLocalProcessingEnergy(task);
                double responseTime = task.calculateResponseTime();
                boolean meetsDeadline = task.meetsDeadline();
                
                // Reward formula: balance between energy efficiency and response time
                reward = calculateReward(energyConsumed, responseTime, meetsDeadline);
                
                // System.out.println("Task " + task.getId() + " processed locally on device " + device.getId());
            } else {
                // Task failed (e.g., due to insufficient battery)
                reward = -10.0; // Penalty for failure
                failedTasks++;
                
                // System.out.println("Task " + task.getId() + " failed to process locally on device " + device.getId());
            }
        } else if (action <= servers.size()) {
            // Offload the task to a server
            int serverIndex = action - 1;
            EdgeServer server = servers.get(serverIndex);
            
            // Calculate bandwidth allocation (simplified)
            double bandwidth = server.getMaxBandwidth() / (server.getDeviceBandwidthMap().size() + 1);
            
            // Allocate bandwidth
            boolean bandwidthAllocated = server.allocateBandwidth(device.getId(), bandwidth);
            
            if (bandwidthAllocated) {
                // Offload the task
                double offloadingEnergy = device.offloadTask(task, bandwidth, currentTime);
                
                // Process the task on the server
                boolean success = server.processTask(task, currentTime + task.calculateTransmissionTime(bandwidth));
                
                if (success) {
                    // Calculate reward
                    double serverEnergy = server.calculateProcessingEnergy(task);
                    double responseTime = task.calculateResponseTime();
                    boolean meetsDeadline = task.meetsDeadline();
                    
                    // Reward formula: balance between energy efficiency and response time
                    reward = calculateReward(offloadingEnergy + serverEnergy, responseTime, meetsDeadline);
                    
                    // System.out.println("Task " + task.getId() + " offloaded from device " + device.getId() + " to server " + server.getId());
                } else {
                    // Task failed to process on the server
                    reward = -10.0; // Penalty for failure
                    failedTasks++;
                    
                    // System.out.println("Task " + task.getId() + " failed to process on server " + server.getId());
                }
                
                // Release bandwidth
                server.releaseBandwidth(device.getId());
            } else {
                // Failed to allocate bandwidth
                reward = -5.0; // Penalty for bandwidth allocation failure
                task.setStatus(Task.TaskStatus.FAILED);
                failedTasks++;
                
                // System.out.println("Failed to allocate bandwidth for task " + task.getId() + " on server " + server.getId());
            }
        }
        
        return reward;
    }
    
    /**
     * Calculate the reward for an action based on energy consumption, response time, and deadline.
     * 
     * @param energyConsumed Energy consumed in Joules
     * @param responseTime Response time in seconds
     * @param meetsDeadline Whether the task meets its deadline
     * @return Reward value
     */
    private double calculateReward(double energyConsumed, double responseTime, boolean meetsDeadline) {
        // Normalize energy consumption and response time using configurable parameters
        double normalizedEnergy = Math.min(1.0, energyConsumed / maxEnergyForNormalization);
        double normalizedTime = Math.min(1.0, responseTime / maxTimeForNormalization);
        
        // Calculate reward using configurable weights
        double reward = -energyWeight * normalizedEnergy - timeWeight * normalizedTime;
        
        // Add configurable bonus/penalty for meeting/missing deadline
        if (meetsDeadline) {
            reward += deadlineBonusReward;
        } else {
            reward -= deadlinePenaltyReward;
        }
        
        // Log reward calculation details
        System.out.println("Reward calculation: Energy=" + energyConsumed + "J (normalized=" + normalizedEnergy + 
                         "), Time=" + responseTime + "s (normalized=" + normalizedTime + "), " +
                         "Meets deadline=" + meetsDeadline + ", Reward=" + reward);
        
        return reward;
    }
    
    /**
     * Update the system state after processing tasks.
     */
    private void updateSystemState() {
        // Update device states
        for (IoTDevice device : devices) {
            // Consume idle energy if the device is not processing a task
            if (device.getCurrentTask() == null) {
                device.consumeIdleEnergy(timeStep);
            }
        }
        
        // Update server states
        for (EdgeServer server : servers) {
            // Update tasks being processed
            List<Task> processingTasks = server.getProcessingTasks();
            Iterator<Task> iterator = processingTasks.iterator();
            
            while (iterator.hasNext()) {
                Task task = iterator.next();
                
                // Check if the task has finished processing
                if (currentTime >= task.getFinishTime()) {
                    server.completeTask(task);
                    completedTasks++;
                    
                    // Update metrics
                    totalResponseTime += task.calculateResponseTime();
                    if (!task.meetsDeadline()) {
                        totalDeadlineMissRate += 1.0;
                    }
                    
                    // System.out.println("Task " + task.getId() + " completed on server " + server.getId());
                }
            }
            
            // Consume idle energy if the server has no tasks
            if (processingTasks.isEmpty()) {
                server.consumeIdleEnergy(timeStep);
            }
        }
        
        // Update total energy consumed
        double energyConsumed = 0.0;
        for (IoTDevice device : devices) {
            energyConsumed += device.getTotalEnergyConsumed();
        }
        for (EdgeServer server : servers) {
            energyConsumed += server.getTotalEnergyConsumed();
        }
        totalEnergyConsumed = energyConsumed;
    }
    
    /**
     * Record metrics for analysis.
     */
    private void recordMetrics() {
        // Record energy consumption
        energyConsumptionHistory.add(totalEnergyConsumed);
        
        // Record response time (average of completed tasks)
        double avgResponseTime = completedTasks > 0 ? totalResponseTime / completedTasks : 0.0;
        responseTimeHistory.add(avgResponseTime);
        
        // Record deadline miss rate
        double deadlineMissRate = totalTasks > 0 ? (totalDeadlineMissRate / totalTasks) * 100 : 0.0;
        deadlineMissRateHistory.add(deadlineMissRate);
        
        // Record task completion rate
        double taskCompletionRate = totalTasks > 0 ? (double) completedTasks / totalTasks * 100 : 0.0;
        taskCompletionRateHistory.add(taskCompletionRate);
    }
    
    /**
     * Get the energy consumption history.
     * 
     * @return List of energy consumption values
     */
    public List<Double> getEnergyConsumptionHistory() {
        return new ArrayList<>(energyConsumptionHistory);
    }
    
    /**
     * Get the response time history.
     * 
     * @return List of response time values
     */
    public List<Double> getResponseTimeHistory() {
        return new ArrayList<>(responseTimeHistory);
    }
    
    /**
     * Get the deadline miss rate history.
     * 
     * @return List of deadline miss rate values
     */
    public List<Double> getDeadlineMissRateHistory() {
        return new ArrayList<>(deadlineMissRateHistory);
    }
    
    /**
     * Get the task completion rate history.
     * 
     * @return List of task completion rate values
     */
    public List<Double> getTaskCompletionRateHistory() {
        return new ArrayList<>(taskCompletionRateHistory);
    }
    
    /**
     * Get the total number of tasks.
     * 
     * @return Total number of tasks
     */
    public int getTotalTasks() {
        return totalTasks;
    }
    
    /**
     * Get the number of completed tasks.
     * 
     * @return Number of completed tasks
     */
    public int getCompletedTasks() {
        return completedTasks;
    }
    
    /**
     * Get the number of failed tasks.
     * 
     * @return Number of failed tasks
     */
    public int getFailedTasks() {
        return failedTasks;
    }
    
    /**
     * Get the total energy consumed.
     * 
     * @return Total energy consumed in Joules
     */
    public double getTotalEnergyConsumed() {
        return totalEnergyConsumed;
    }
    
    /**
     * Get the average response time.
     * 
     * @return Average response time in seconds
     */
    public double getAverageResponseTime() {
        return completedTasks > 0 ? totalResponseTime / completedTasks : 0.0;
    }
    
    /**
     * Get the deadline miss rate.
     * 
     * @return Deadline miss rate as a percentage
     */
    public double getDeadlineMissRate() {
        return totalTasks > 0 ? (totalDeadlineMissRate / totalTasks) * 100 : 0.0;
    }
    
    /**
     * Get the task completion rate.
     * 
     * @return Task completion rate as a percentage
     */
    public double getTaskCompletionRate() {
        return totalTasks > 0 ? (double) completedTasks / totalTasks * 100 : 0.0;
    }
}
