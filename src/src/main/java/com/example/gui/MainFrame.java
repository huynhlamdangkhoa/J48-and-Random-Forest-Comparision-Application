package com.example.gui;

import javax.swing.JFrame;
import com.example.controllers.MiningController;

public class MainFrame extends JFrame {
    
    public static void main(String[] args) {
        try {          
            // Initialize controller
            MiningController controller = new MiningController();
            
            controller.runPipeline(
                "src/resources/heart_disease.csv",    // rawPath: Input dataset
                "src/resources/evaluation_report.txt" // reportPath: Output report
            );
            
            
        } catch (Exception e) {
            System.err.println("\nERROR: Pipeline execution failed!");
            System.err.println("Error message: " + e.getMessage());
            System.err.println("\nStack trace:");
            e.printStackTrace();
            
            System.exit(1);
        }
    }
}