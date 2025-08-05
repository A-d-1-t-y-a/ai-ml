package org.jcora.mec.config;

import org.jcora.mec.model.EdgeServer;
import org.jcora.mec.model.IoTDevice;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/**
 * Utility class for loading simulation configuration from properties files.
 */
public class ConfigurationLoader {
    private static final Logger logger = LoggerFactory.getLogger(ConfigurationLoader.class);
    
    // Default configuration file path
    private static final String DEFAULT_CONFIG_PATH = "config/simulation.properties";
    
    // Configuration properties
    private final Properties properties;
    
    /**
     * Constructor for creating a new configuration loader with the default configuration file.
     */
    public ConfigurationLoader() {
        this(DEFAULT_CONFIG_PATH);
    }
    
    /**
     * Constructor for creating a new configuration loader with a specified configuration file.
     * 
     * @param configPath Path to the configuration file
     */
    public ConfigurationLoader(String configPath) {
        properties = new Properties();
        try (InputStream input = new FileInputStream(configPath)) {
            properties.load(input);
            logger.info("Loaded configuration from {}", configPath);
        } catch (IOException e) {
            logger.error("Failed to load configuration from {}: {}", configPath, e.getMessage());
        }
    }
    
    /**
     * Get a string property from the configuration.
     * 
     * @param key Property key
     * @param defaultValue Default value if the property is not found
     * @return Property value
     */
    public String getString(String key, String defaultValue) {
        return properties.getProperty(key, defaultValue);
    }
    
    /**
     * Get an integer property from the configuration.
     * 
     * @param key Property key
     * @param defaultValue Default value if the property is not found
     * @return Property value
     */
    public int getInt(String key, int defaultValue) {
        String value = properties.getProperty(key);
        if (value == null) {
            return defaultValue;
        }
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            logger.warn("Invalid integer value for {}: {}", key, value);
            return defaultValue;
        }
    }
    
    /**
     * Get a double property from the configuration.
     * 
     * @param key Property key
     * @param defaultValue Default value if the property is not found
     * @return Property value
     */
    public double getDouble(String key, double defaultValue) {
        String value = properties.getProperty(key);
        if (value == null) {
            return defaultValue;
        }
        try {
            return Double.parseDouble(value);
        } catch (NumberFormatException e) {
            logger.warn("Invalid double value for {}: {}", key, value);
            return defaultValue;
        }
    }
    
    /**
     * Get a boolean property from the configuration.
     * 
     * @param key Property key
     * @param defaultValue Default value if the property is not found
     * @return Property value
     */
    public boolean getBoolean(String key, boolean defaultValue) {
        String value = properties.getProperty(key);
        if (value == null) {
            return defaultValue;
        }
        return Boolean.parseBoolean(value);
    }
    
    /**
     * Create IoT devices based on the configuration.
     * 
     * @return List of IoT devices
     */
    public List<IoTDevice> createIoTDevices() {
        List<IoTDevice> devices = new ArrayList<>();
        
        int numDevices = getInt("simulation.devices.count", 5);
        
        for (int i = 0; i < numDevices; i++) {
            // Get device-specific properties or use defaults
            String name = getString("simulation.device." + i + ".name", "Device-" + i);
            double processingPower = getDouble("simulation.device." + i + ".processingPower", 
                    getDouble("simulation.devices.default.processingPower", 500.0));
            double energyConsumption = getDouble("simulation.device." + i + ".energyConsumption", 
                    getDouble("simulation.devices.default.energyConsumption", 2.0));
            double idleEnergyConsumption = getDouble("simulation.device." + i + ".idleEnergyConsumption", 
                    getDouble("simulation.devices.default.idleEnergyConsumption", 0.5));
            double transmissionPower = getDouble("simulation.device." + i + ".transmissionPower", 
                    getDouble("simulation.devices.default.transmissionPower", 1.0));
            double batteryCapacity = getDouble("simulation.device." + i + ".batteryCapacity", 
                    getDouble("simulation.devices.default.batteryCapacity", 5000.0));
            
            // Create device
            IoTDevice device = new IoTDevice(i, name, processingPower, energyConsumption, 
                    idleEnergyConsumption, transmissionPower, batteryCapacity);
            devices.add(device);
            
            logger.debug("Created IoT device: {}", device);
        }
        
        return devices;
    }
    
    /**
     * Create edge servers based on the configuration.
     * 
     * @return List of edge servers
     */
    public List<EdgeServer> createEdgeServers() {
        List<EdgeServer> servers = new ArrayList<>();
        
        int numServers = getInt("simulation.servers.count", 3);
        
        for (int i = 0; i < numServers; i++) {
            // Get server-specific properties or use defaults
            String name = getString("simulation.server." + i + ".name", "Server-" + i);
            double processingPower = getDouble("simulation.server." + i + ".processingPower", 
                    getDouble("simulation.servers.default.processingPower", 2000.0));
            double energyConsumption = getDouble("simulation.server." + i + ".energyConsumption", 
                    getDouble("simulation.servers.default.energyConsumption", 5.0));
            double idleEnergyConsumption = getDouble("simulation.server." + i + ".idleEnergyConsumption", 
                    getDouble("simulation.servers.default.idleEnergyConsumption", 2.0));
            double maxBandwidth = getDouble("simulation.server." + i + ".maxBandwidth", 
                    getDouble("simulation.servers.default.maxBandwidth", 100.0));
            int maxConnections = getInt("simulation.server." + i + ".maxConnections", 
                    getInt("simulation.servers.default.maxConnections", 10));
            
            // Create server
            EdgeServer server = new EdgeServer(i, name, processingPower, energyConsumption, 
                    idleEnergyConsumption, maxBandwidth, maxConnections);
            servers.add(server);
            
            logger.debug("Created edge server: {}", server);
        }
        
        return servers;
    }
    
    /**
     * Get the simulation duration from the configuration.
     * 
     * @return Simulation duration in seconds
     */
    public double getSimulationDuration() {
        return getDouble("simulation.duration", 100.0);
    }
    
    /**
     * Get the simulation time step from the configuration.
     * 
     * @return Simulation time step in seconds
     */
    public double getTimeStep() {
        return getDouble("simulation.timeStep", 1.0);
    }
    
    /**
     * Get the task generation probability from the configuration.
     * 
     * @return Task generation probability
     */
    public double getTaskGenerationProbability() {
        return getDouble("simulation.taskGenerationProbability", 0.2);
    }
    
    /**
     * Get the output directory from the configuration.
     * 
     * @return Output directory path
     */
    public String getOutputDirectory() {
        return getString("simulation.outputDir", "output");
    }
    
    /**
     * Get the scenario name from the configuration.
     * 
     * @return Scenario name
     */
    public String getScenarioName() {
        return getString("simulation.scenarioName", "default");
    }
    
    /**
     * Get the DRL agent parameters from the configuration.
     * 
     * @return Array of DRL agent parameters [stateSize, actionSize, gamma, epsilon, epsilonMin, epsilonDecay, batchSize, replayMemorySize, targetNetworkUpdateFreq]
     */
    public Object[] getDRLAgentParameters() {
        int stateSize = getInt("drl.stateSize", 10);
        int actionSize = getInt("drl.actionSize", 4);
        double gamma = getDouble("drl.gamma", 0.95);
        double epsilon = getDouble("drl.epsilon", 1.0);
        double epsilonMin = getDouble("drl.epsilonMin", 0.01);
        double epsilonDecay = getDouble("drl.epsilonDecay", 0.995);
        int batchSize = getInt("drl.batchSize", 32);
        int replayMemorySize = getInt("drl.replayMemorySize", 1000);
        int targetNetworkUpdateFreq = getInt("drl.targetNetworkUpdateFreq", 100);
        
        return new Object[]{stateSize, actionSize, gamma, epsilon, epsilonMin, epsilonDecay, 
                           batchSize, replayMemorySize, targetNetworkUpdateFreq};
    }
}
