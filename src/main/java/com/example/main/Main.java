package com.example.main;

import javax.swing.JFrame;

import com.example.controllers.MiningController;

/**
 * Main entry point cho Heart Disease Prediction System
 */
public class Main {   // ← Không extends JFrame nữa

    public static void main(String[] args) {
        try {
            System.out.println("╔════════════════════════════════════════════════════════╗");
            System.out.println("║    HEART DISEASE RISK PREDICTOR - DATA MINING SYSTEM   ║");
            System.out.println("╚════════════════════════════════════════════════════════╝");

            MiningController controller = new MiningController();

            controller.runPipeline(
                    "src/main/resources/heart_disease.csv",
                    "src/main/resources/evaluation_report.txt"
            );

            System.out.println("\n╔════════════════════════════════════════════════════════╗");
            System.out.println("║          PIPELINE COMPLETED SUCCESSFULLY!          ║");
            System.out.println("╚════════════════════════════════════════════════════════╝");

        } catch (Exception e) {
            System.err.println("\nERROR: Pipeline execution failed!");
            System.err.println("Message: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}