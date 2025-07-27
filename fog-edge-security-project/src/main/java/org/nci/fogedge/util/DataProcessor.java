package org.nci.fogedge.util;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Utility class for data processing operations
 * Based on the paper: "An Overview of Fog Computing and Edge Computing Security and Privacy Issues"
 * (Sensors 2021, 21, 8226, https://doi.org/10.3390/s21248226)
 */
public class DataProcessor {
    private static final Logger logger = LogManager.getLogger(DataProcessor.class);
    
    /**
     * Encrypt data using a simple encryption algorithm
     * In a real implementation, this would use proper cryptographic libraries
     * @param data The data to encrypt
     * @return The encrypted data
     */
    public static byte[] encryptData(byte[] data) {
        if (data == null || data.length == 0) {
            return data;
        }
        
        // Simple XOR encryption (for demonstration only)
        byte[] encrypted = new byte[data.length];
        byte key = 42; // Simple key
        
        for (int i = 0; i < data.length; i++) {
            encrypted[i] = (byte) (data[i] ^ key);
        }
        
        logger.debug("Encrypted {} bytes of data", data.length);
        return encrypted;
    }
    
    /**
     * Decrypt data using a simple decryption algorithm
     * In a real implementation, this would use proper cryptographic libraries
     * @param data The data to decrypt
     * @return The decrypted data
     */
    public static byte[] decryptData(byte[] data) {
        if (data == null || data.length == 0) {
            return data;
        }
        
        // Simple XOR decryption (for demonstration only)
        byte[] decrypted = new byte[data.length];
        byte key = 42; // Same key as encryption
        
        for (int i = 0; i < data.length; i++) {
            decrypted[i] = (byte) (data[i] ^ key);
        }
        
        logger.debug("Decrypted {} bytes of data", data.length);
        return decrypted;
    }
    
    /**
     * Verify data integrity using a simple checksum
     * In a real implementation, this would use proper cryptographic hash functions
     * @param data The data to verify
     * @param checksum The checksum to verify against
     * @return True if the data integrity is verified
     */
    public static boolean verifyIntegrity(byte[] data, int checksum) {
        if (data == null) {
            return false;
        }
        
        int calculatedChecksum = calculateChecksum(data);
        boolean verified = calculatedChecksum == checksum;
        
        logger.debug("Data integrity verification: {}", verified);
        return verified;
    }
    
    /**
     * Calculate a simple checksum for data integrity
     * In a real implementation, this would use proper cryptographic hash functions
     * @param data The data to calculate checksum for
     * @return The calculated checksum
     */
    public static int calculateChecksum(byte[] data) {
        if (data == null) {
            return 0;
        }
        
        int checksum = 0;
        for (byte b : data) {
            checksum = (checksum + (b & 0xFF)) % 65536;
        }
        
        logger.debug("Calculated checksum: {}", checksum);
        return checksum;
    }
    
    /**
     * Anonymize sensitive data
     * @param data The data to anonymize
     * @return The anonymized data
     */
    public static String anonymizeData(String data) {
        if (data == null || data.isEmpty()) {
            return data;
        }
        
        // Simple anonymization (for demonstration only)
        // In a real implementation, this would use more sophisticated techniques
        String anonymized = data.replaceAll("\\d{4}-\\d{4}-\\d{4}-\\d{4}", "XXXX-XXXX-XXXX-XXXX") // Credit card
                .replaceAll("\\d{3}-\\d{2}-\\d{4}", "XXX-XX-XXXX") // SSN
                .replaceAll("\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\\b", "XXXXX@XXXXX.XXX"); // Email
        
        logger.debug("Anonymized sensitive data");
        return anonymized;
    }
    
    /**
     * Filter out malicious content from data
     * @param data The data to filter
     * @return The filtered data
     */
    public static String filterMaliciousContent(String data) {
        if (data == null || data.isEmpty()) {
            return data;
        }
        
        // Simple filtering (for demonstration only)
        // In a real implementation, this would use more sophisticated techniques
        String filtered = data.replaceAll("(?i)<script.*?>.*?</script>", "") // Remove script tags
                .replaceAll("(?i)javascript:", "") // Remove javascript: protocol
                .replaceAll("(?i)on\\w+\\s*=", "disabled="); // Disable event handlers
        
        logger.debug("Filtered potentially malicious content");
        return filtered;
    }
}
