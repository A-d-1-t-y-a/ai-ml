package org.jcora.mec.drl;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.jcora.mec.model.EdgeServer;
import org.jcora.mec.model.IoTDevice;
import org.jcora.mec.model.Task;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Deep Reinforcement Learning Agent for Joint Computation Offloading and Resource Allocation (JCORA).
 * This agent uses a Deep Q-Network (DQN) to make decisions on task offloading and resource allocation.
 */
public class DRLAgent {
    private static final Logger logger = LoggerFactory.getLogger(DRLAgent.class);
    
    // DRL parameters
    private final double gamma;               // Discount factor
    private double epsilon;                    // Exploration rate (mutable for decay)
    private final double epsilonMin;          // Minimum exploration rate
    private final double epsilonDecay;        // Exploration rate decay
    private final int batchSize;              // Batch size for training
    private final int replayMemorySize;       // Size of replay memory
    private final int targetNetworkUpdateFreq; // Frequency of target network update
    
    // Neural network models
    private MultiLayerNetwork qNetwork;       // Q-Network for action selection
    private MultiLayerNetwork targetNetwork;  // Target network for stable learning
    
    // Replay memory
    private final List<Experience> replayMemory;
    private final Random random;
    
    // State and action dimensions
    private final int stateSize;
    private final int actionSize;
    
    /**
     * Constructor for creating a new DRL Agent.
     * 
     * @param stateSize Dimension of the state space
     * @param actionSize Dimension of the action space
     * @param gamma Discount factor
     * @param epsilon Initial exploration rate
     * @param epsilonMin Minimum exploration rate
     * @param epsilonDecay Exploration rate decay
     * @param batchSize Batch size for training
     * @param replayMemorySize Size of replay memory
     * @param targetNetworkUpdateFreq Frequency of target network update
     */
    public DRLAgent(int stateSize, int actionSize, double gamma, double epsilon, double epsilonMin,
                   double epsilonDecay, int batchSize, int replayMemorySize, int targetNetworkUpdateFreq) {
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        this.gamma = gamma;
        this.epsilon = epsilon;
        this.epsilonMin = epsilonMin;
        this.epsilonDecay = epsilonDecay;
        this.batchSize = batchSize;
        this.replayMemorySize = replayMemorySize;
        this.targetNetworkUpdateFreq = targetNetworkUpdateFreq;
        
        this.replayMemory = new ArrayList<>(replayMemorySize);
        this.random = new Random();
        
        // Initialize neural networks
        initializeNetworks();
    }
    
    /**
     * Initialize the Q-Network and Target Network.
     */
    private void initializeNetworks() {
        // Define neural network configuration
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.001))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(stateSize)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(64)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(64)
                        .nOut(actionSize)
                        .activation(Activation.IDENTITY)
                        .build())
                .build();
        
        // Initialize Q-Network
        qNetwork = new MultiLayerNetwork(config);
        qNetwork.init();
        
        // Initialize Target Network with same parameters
        targetNetwork = new MultiLayerNetwork(config);
        targetNetwork.init();
        
        // Copy weights from Q-Network to Target Network
        targetNetwork.setParams(qNetwork.params().dup());
        
        logger.info("Neural networks initialized successfully");
    }
    
    /**
     * Get the state representation for the current environment.
     * 
     * @param device IoT device
     * @param servers List of edge servers
     * @param task Current task
     * @return State vector as INDArray
     */
    public INDArray getState(IoTDevice device, List<EdgeServer> servers, Task task) {
        // Create state vector
        double[] state = new double[stateSize];
        int index = 0;
        
        // Device state features
        state[index++] = device.getProcessingPower() / 1000.0; // Normalize
        state[index++] = device.getRemainingBattery() / device.getBatteryCapacity(); // Battery percentage
        
        // Task features
        state[index++] = task.getInputDataSize() / 100.0; // Normalize
        state[index++] = task.getOutputDataSize() / 100.0; // Normalize
        state[index++] = task.getComputationalRequirement() / 10000.0; // Normalize
        state[index++] = task.getDeadline() / 10.0; // Normalize
        
        // Server features (for each server)
        for (EdgeServer server : servers) {
            state[index++] = server.getCurrentLoad() / 100.0; // Load percentage
            state[index++] = server.getAllocatedBandwidth(device.getId()) / server.getMaxBandwidth(); // Bandwidth allocation
        }
        
        // Convert to INDArray
        return Nd4j.create(state);
    }
    
    /**
     * Select an action based on the current state using epsilon-greedy policy.
     * 
     * @param state Current state
     * @return Selected action index
     */
    public int selectAction(INDArray state) {
        // Epsilon-greedy action selection
        if (random.nextDouble() < epsilon) {
            // Exploration: select a random action
            return random.nextInt(actionSize);
        } else {
            // Exploitation: select the best action according to the Q-Network
            INDArray qValues = qNetwork.output(state);
            return Nd4j.argMax(qValues, 1).getInt(0);
        }
    }
    
    /**
     * Store experience in replay memory.
     * 
     * @param state Current state
     * @param action Selected action
     * @param reward Received reward
     * @param nextState Next state
     * @param done Whether the episode is done
     */
    public void remember(INDArray state, int action, double reward, INDArray nextState, boolean done) {
        // Create new experience
        Experience experience = new Experience(state, action, reward, nextState, done);
        
        // Add to replay memory
        if (replayMemory.size() >= replayMemorySize) {
            replayMemory.remove(0); // Remove oldest experience if memory is full
        }
        replayMemory.add(experience);
    }
    
    /**
     * Train the agent using experiences from replay memory.
     * 
     * @param updateStep Current update step
     */
    public void train(int updateStep) {
        // Check if there are enough experiences in replay memory
        if (replayMemory.size() < batchSize) {
            return;
        }
        
        // Sample a batch of experiences from replay memory
        List<Experience> batch = sampleBatch();
        
        // Prepare training data
        INDArray states = Nd4j.create(batchSize, stateSize);
        INDArray targets = Nd4j.create(batchSize, actionSize);
        
        // Calculate target Q-values for each experience in the batch
        for (int i = 0; i < batchSize; i++) {
            Experience experience = batch.get(i);
            
            // Get current Q-values
            INDArray qValues = qNetwork.output(experience.getState());
            
            // Copy Q-values to targets
            targets.putRow(i, qValues);
            
            // Calculate target Q-value for the selected action
            double targetQ;
            if (experience.isDone()) {
                // If episode is done, target is just the reward
                targetQ = experience.getReward();
            } else {
                // Otherwise, target is reward + gamma * max(Q(s', a'))
                INDArray nextQValues = targetNetwork.output(experience.getNextState());
                targetQ = experience.getReward() + gamma * nextQValues.maxNumber().doubleValue();
            }
            
            // Update target for the selected action
            targets.putScalar(new int[]{i, experience.getAction()}, targetQ);
        }
        
        // Train the Q-Network
        qNetwork.fit(states, targets);
        
        // Update target network if needed
        if (updateStep % targetNetworkUpdateFreq == 0) {
            targetNetwork.setParams(qNetwork.params().dup());
            logger.info("Target network updated at step {}", updateStep);
        }
        
        // Decay epsilon
        if (epsilon > epsilonMin) {
            epsilon *= epsilonDecay;
        }
    }
    
    /**
     * Sample a batch of experiences from replay memory.
     * 
     * @return List of sampled experiences
     */
    private List<Experience> sampleBatch() {
        List<Experience> batch = new ArrayList<>(batchSize);
        
        // Randomly sample experiences from replay memory
        for (int i = 0; i < batchSize; i++) {
            int index = random.nextInt(replayMemory.size());
            batch.add(replayMemory.get(index));
        }
        
        return batch;
    }
    
    /**
     * Save the trained model to a file.
     * 
     * @param filePath Path to save the model
     */
    public void saveModel(String filePath) {
        try {
            File file = new File(filePath);
            qNetwork.save(file);
            logger.info("Model saved to {}", filePath);
        } catch (IOException e) {
            logger.error("Failed to save model: {}", e.getMessage());
        }
    }
    
    /**
     * Load a trained model from a file.
     * 
     * @param filePath Path to load the model from
     */
    public void loadModel(String filePath) {
        try {
            File file = new File(filePath);
            qNetwork = MultiLayerNetwork.load(file, true);
            targetNetwork.setParams(qNetwork.params().dup());
            logger.info("Model loaded from {}", filePath);
        } catch (IOException e) {
            logger.error("Failed to load model: {}", e.getMessage());
        }
    }
    
    /**
     * Inner class representing an experience for replay memory.
     */
    private static class Experience {
        private final INDArray state;
        private final int action;
        private final double reward;
        private final INDArray nextState;
        private final boolean done;
        
        public Experience(INDArray state, int action, double reward, INDArray nextState, boolean done) {
            this.state = state;
            this.action = action;
            this.reward = reward;
            this.nextState = nextState;
            this.done = done;
        }
        
        public INDArray getState() {
            return state;
        }
        
        public int getAction() {
            return action;
        }
        
        public double getReward() {
            return reward;
        }
        
        public INDArray getNextState() {
            return nextState;
        }
        
        public boolean isDone() {
            return done;
        }
    }
}
