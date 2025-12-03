package com.example.main;

import javax.swing.JFrame;

import com.example.controllers.MiningController;

import java.util.Locale;

/**
 * Main entry point cho Heart Disease Prediction System
 */
public class Main {

    public static void main(String[] args) {
        Locale.setDefault(Locale.US);
        try {
            System.out.println("HEART DISEASE RISK PREDICTOR");

            MiningController controller = new MiningController();

            controller.runPipeline(
                    "src/main/resources/heart_disease.csv",
                    "src/main/resources/evaluation_report.txt"
            );

            System.out.println("PIPELINE COMPLETED SUCCESSFULLY!");


        } catch (Exception e) {
            System.err.println("\nERROR: Pipeline execution failed!");
            System.err.println("Message: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}