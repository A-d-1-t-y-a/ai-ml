package com.nci.fogedge.devices;

/**
 * Enum representing different wireless communication technologies used by IoT devices.
 */
public enum WirelessType {
    /**
     * WiFi (IEEE 802.11) - High bandwidth, medium range, high power consumption
     */
    WIFI(100.0, 50.0, 100.0),
    
    /**
     * Bluetooth (IEEE 802.15.1) - Medium bandwidth, short range, low power consumption
     */
    BLUETOOTH(3.0, 10.0, 15.0),
    
    /**
     * ZigBee (IEEE 802.15.4) - Low bandwidth, medium range, very low power consumption
     */
    ZIGBEE(0.25, 30.0, 5.0),
    
    /**
     * Cellular (4G/5G) - High bandwidth, long range, high power consumption
     */
    CELLULAR(50.0, 1000.0, 200.0),
    
    /**
     * LoRaWAN - Very low bandwidth, very long range, extremely low power consumption
     */
    LORAWAN(0.05, 5000.0, 1.0),
    
    /**
     * NB-IoT - Low bandwidth, long range, low power consumption
     */
    NBIOT(0.2, 1000.0, 10.0);
    
    private final double bandwidthMbps; // Bandwidth in Mbps
    private final double rangeMeters; // Range in meters
    private final double powerConsumptionMa; // Power consumption in mA
    
    /**
     * Constructor for WirelessType
     * 
     * @param bandwidthMbps Bandwidth in Mbps
     * @param rangeMeters Range in meters
     * @param powerConsumptionMa Power consumption in mA
     */
    WirelessType(double bandwidthMbps, double rangeMeters, double powerConsumptionMa) {
        this.bandwidthMbps = bandwidthMbps;
        this.rangeMeters = rangeMeters;
        this.powerConsumptionMa = powerConsumptionMa;
    }
    
    /**
     * Gets the bandwidth of this wireless type
     * 
     * @return Bandwidth in Mbps
     */
    public double getBandwidthMbps() {
        return bandwidthMbps;
    }
    
    /**
     * Gets the range of this wireless type
     * 
     * @return Range in meters
     */
    public double getRangeMeters() {
        return rangeMeters;
    }
    
    /**
     * Gets the power consumption of this wireless type
     * 
     * @return Power consumption in mA
     */
    public double getPowerConsumptionMa() {
        return powerConsumptionMa;
    }
}
