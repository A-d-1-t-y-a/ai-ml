package com.nci.fogedge.network;

/**
 * Network Location for Fog and Edge Computing System
 * 
 * This class represents geographical coordinates for network nodes.
 * It provides distance calculation and location-based routing capabilities.
 * 
 * @author National College of Ireland - Fog and Edge Computing Project
 * @version 1.0.0
 * @since 2024
 */
public class NetworkLocation {
    
    private final double latitude;
    private final double longitude;
    private final double altitude;
    
    /**
     * Constructor for NetworkLocation
     * 
     * @param latitude Latitude coordinate
     * @param longitude Longitude coordinate
     * @param altitude Altitude in meters
     */
    public NetworkLocation(double latitude, double longitude, double altitude) {
        this.latitude = latitude;
        this.longitude = longitude;
        this.altitude = altitude;
    }
    
    /**
     * Constructor for NetworkLocation (ground level)
     * 
     * @param latitude Latitude coordinate
     * @param longitude Longitude coordinate
     */
    public NetworkLocation(double latitude, double longitude) {
        this(latitude, longitude, 0.0);
    }
    
    /**
     * Get latitude coordinate
     * 
     * @return Latitude
     */
    public double getLatitude() {
        return latitude;
    }
    
    /**
     * Get longitude coordinate
     * 
     * @return Longitude
     */
    public double getLongitude() {
        return longitude;
    }
    
    /**
     * Get altitude
     * 
     * @return Altitude in meters
     */
    public double getAltitude() {
        return altitude;
    }
    
    /**
     * Calculate distance to another location using Haversine formula
     * 
     * @param other Other location
     * @return Distance in meters
     */
    public double calculateDistance(NetworkLocation other) {
        if (other == null) {
            return Double.MAX_VALUE;
        }
        
        // Haversine formula for great circle distance
        double lat1 = Math.toRadians(this.latitude);
        double lat2 = Math.toRadians(other.latitude);
        double deltaLat = Math.toRadians(other.latitude - this.latitude);
        double deltaLon = Math.toRadians(other.longitude - this.longitude);
        
        double a = Math.sin(deltaLat / 2) * Math.sin(deltaLat / 2) +
                   Math.cos(lat1) * Math.cos(lat2) *
                   Math.sin(deltaLon / 2) * Math.sin(deltaLon / 2);
        
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        
        // Earth's radius in meters
        double earthRadius = 6371000;
        
        // Calculate horizontal distance
        double horizontalDistance = earthRadius * c;
        
        // Add altitude difference
        double altitudeDifference = Math.abs(this.altitude - other.altitude);
        
        // Calculate total distance (Pythagorean theorem)
        return Math.sqrt(horizontalDistance * horizontalDistance + altitudeDifference * altitudeDifference);
    }
    
    /**
     * Check if location is within range of another location
     * 
     * @param other Other location
     * @param rangeMeters Range in meters
     * @return True if within range
     */
    public boolean isWithinRange(NetworkLocation other, double rangeMeters) {
        return calculateDistance(other) <= rangeMeters;
    }
    
    /**
     * Create a random location within specified bounds
     * 
     * @param minLat Minimum latitude
     * @param maxLat Maximum latitude
     * @param minLon Minimum longitude
     * @param maxLon Maximum longitude
     * @return Random location
     */
    public static NetworkLocation randomLocation(double minLat, double maxLat, 
                                              double minLon, double maxLon) {
        double lat = minLat + Math.random() * (maxLat - minLat);
        double lon = minLon + Math.random() * (maxLon - minLon);
        double alt = Math.random() * 100; // Random altitude 0-100m
        
        return new NetworkLocation(lat, lon, alt);
    }
    
    /**
     * Create a random location in Dublin area
     * 
     * @return Random location in Dublin
     */
    public static NetworkLocation randomDublinLocation() {
        // Dublin coordinates: 53.3498° N, 6.2603° W
        double dublinLat = 53.3498;
        double dublinLon = -6.2603;
        double range = 0.1; // ~10km range
        
        double lat = dublinLat + (Math.random() - 0.5) * range;
        double lon = dublinLon + (Math.random() - 0.5) * range;
        double alt = Math.random() * 50; // 0-50m altitude
        
        return new NetworkLocation(lat, lon, alt);
    }
    
    @Override
    public String toString() {
        return String.format("NetworkLocation{lat=%.6f, lon=%.6f, alt=%.1fm}",
            latitude, longitude, altitude);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        NetworkLocation that = (NetworkLocation) obj;
        return Double.compare(that.latitude, latitude) == 0 &&
               Double.compare(that.longitude, longitude) == 0 &&
               Double.compare(that.altitude, altitude) == 0;
    }
    
    @Override
    public int hashCode() {
        int result = 17;
        result = 31 * result + Double.hashCode(latitude);
        result = 31 * result + Double.hashCode(longitude);
        result = 31 * result + Double.hashCode(altitude);
        return result;
    }
} 