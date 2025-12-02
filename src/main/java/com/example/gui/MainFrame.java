package com.example.gui;

import javax.swing.JFrame;

import com.example.controllers.MiningController;

/**
 * Main entry point cho Heart Disease Prediction System
 */
public class MainFrame extends JFrame {
    
    public static void main(String[] args) {
        try {
            System.out.println("╔════════════════════════════════════════════════════════╗");
            System.out.println("║   HEART DISEASE RISK PREDICTOR - DATA MINING SYSTEM   ║");
            System.out.println("╚════════════════════════════════════════════════════════╝");
            
            // Initialize controller
            MiningController controller = new MiningController();
            
            // ✅ ĐÚNG - Chỉ 2 parameters
            controller.runPipeline(
                "src/main/resources/heart_disease.csv",    // rawPath: Input dataset
                "src/main/resources/evaluation_report.txt" // reportPath: Output report
            );
            
            System.out.println("\n╔════════════════════════════════════════════════════════╗");
            System.out.println("║              🎉 PIPELINE COMPLETED! 🎉                 ║");
            System.out.println("╚════════════════════════════════════════════════════════╝");
            
        } catch (Exception e) {
            System.err.println("\n❌ ERROR: Pipeline execution failed!");
            System.err.println("Error message: " + e.getMessage());
            System.err.println("\nStack trace:");
            e.printStackTrace();
            
            System.exit(1);
        }
    }
}