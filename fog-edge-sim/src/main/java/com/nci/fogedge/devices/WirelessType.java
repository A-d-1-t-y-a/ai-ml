package com.nci.fogedge.devices;

/**
 * Enum representing different types of wireless connections for IoT devices.
 * Each type has different characteristics in terms of range, bandwidth, and energy consumption.
 */
public enum WirelessType {
    /**
     * WiFi connection with high bandwidth but high energy consumption.
     * Range: Medium (50-100m)
     * Bandwidth: High (10-100 Mbps)
     * Energy Consumption: High
     */
    WIFI(100.0, 50.0, 0.1),
    
    /**
     * Bluetooth connection with low bandwidth and low energy consumption.
     * Range: Low (10-30m)
     * Bandwidth: Low (1-3 Mbps)
     * Energy Consumption: Low
     */
    BLUETOOTH(30.0, 2.0, 0.02),
    
    /**
     * Zigbee connection with very low bandwidth but very low energy consumption.
     * Range: Low (10-100m)
     * Bandwidth: Very Low (0.02-0.25 Mbps)
     * Energy Consumption: Very Low
     */
    ZIGBEE(100.0, 0.25, 0.01),
    
    /**
     * LoRaWAN connection with very low bandwidth but very long range.
     * Range: Very High (2-15km)
     * Bandwidth: Extremely Low (0.0005-0.05 Mbps)
     * Energy Consumption: Very Low
     */
    LORAWAN(15000.0, 0.05, 0.005),
    
    /**
     * 5G connection with very high bandwidth but high energy consumption.
     * Range: High (300-500m)
     * Bandwidth: Very High (100-1000 Mbps)
     * Energy Consumption: High
     */
    CELLULAR_5G(500.0, 1000.0, 0.15),
    
    /**
     * 4G connection with high bandwidth and medium energy consumption.
     * Range: High (1-10km)
     * Bandwidth: High (5-50 Mbps)
     * Energy Consumption: Medium
     */
    CELLULAR_4G(10000.0, 50.0, 0.1),
    
    /**
     * NB-IoT connection with low bandwidth but low energy consumption.
     * Range: High (1-10km)
     * Bandwidth: Low (0.1-0.2 Mbps)
     * Energy Consumption: Low
     */
    NB_IOT(10000.0, 0.2, 0.02);
    
    private final double range; // in meters
    private final double bandwidth; // in Mbps
    private final double energyConsumptionFactor; // relative factor (0-1)
    
    /**
     * Constructor for WirelessType
     * 
     * @param range Range in meters
     * @param bandwidth Bandwidth in Mbps
     * @param energyConsumptionFactor Energy consumption factor (0-1)
     */
    WirelessType(double range, double bandwidth, double energyConsumptionFactor) {
        this.range = range;
        this.bandwidth = bandwidth;
        this.energyConsumptionFactor = energyConsumptionFactor;
    }
    
    /**
     * Gets the range of the wireless connection
     * 
     * @return Range in meters
     */
    public double getRange() {
        return range;
    }
    
    /**
     * Gets the bandwidth of the wireless connection
     * 
     * @return Bandwidth in Mbps
     */
    public double getBandwidth() {
        return bandwidth;
    }
    
    /**
     * Gets the energy consumption factor of the wireless connection
     * 
     * @return Energy consumption factor (0-1)
     */
    public double getEnergyConsumptionFactor() {
        return energyConsumptionFactor;
    }
}
