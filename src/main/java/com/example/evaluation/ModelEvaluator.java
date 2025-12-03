package com.example.evaluation;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import weka.core.Utils;
import weka.gui.visualize.*;
import javax.swing.*;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.Instances;


public class ModelEvaluator {
    //Store evaluation results for comparison
    private List<EvaluationResult> results = new ArrayList<>();
    private static class EvaluationResult {
        String modelName;
        Evaluation evaluation;
        double runtime;
        Instances data;
        EvaluationResult(String modelName, Evaluation eval, double runtime, Instances data) {
            this.modelName = modelName;
            this.evaluation = eval;
            this.runtime = runtime;
            this.data = data;
        }
    }

    /**
     Evaluate model - Compatible với J48Classifier và RandomForestClassifier
     @param classifier Classifier đã được train (J48, RandomForest, etc.)
     @param data Dataset để evaluate
     @param reportPath Path để save report
     @throws Exception
     */
    public void evaluateModel(Classifier classifier, Instances data, String reportPath)
            throws Exception {
        String modelName = classifier.getClass().getSimpleName();
        System.out.println("\n" + "=".repeat(60));
        System.out.println("EVALUATING: " + modelName);
        System.out.println("=".repeat(60));
        //Measure training time
        long startTime = System.nanoTime();
        //Build classifier
        classifier.buildClassifier(data);
        //10-fold cross-validation
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(classifier, data, 10, new Random(1));
        showROCCurve(classifier.getClass().getSimpleName(), eval);
        long endTime = System.nanoTime();
        double runtime = (endTime - startTime) / 1_000_000_000.0;
        //Print results to console
        printEvaluationResults(modelName, eval, runtime);
        //Save to report file
        saveEvaluationReport(modelName, eval, runtime, reportPath);
        //Store for comparison
        results.add(new EvaluationResult(modelName, eval, runtime, data));
        System.out.println("\n" + modelName + " evaluation completed!");
    }

    private void printEvaluationResults(String modelName, Evaluation eval, double runtime) {
        System.out.println("\nPerformance Metrics:");
        System.out.printf("-Accuracy:           %.2f%%\n", eval.pctCorrect());
        System.out.printf("-Precision (Weighted): %.3f\n", eval.weightedPrecision());
        System.out.printf("-Recall (Weighted):    %.3f\n", eval.weightedRecall());
        System.out.printf("-F1-Score (Weighted):  %.3f\n", eval.weightedFMeasure());

        //Disease class metrics (class index 1)
        try {
            System.out.println("\nDisease Class Metrics (Class 1):");
            System.out.printf("-Precision:  %.3f\n", eval.precision(1));
            System.out.printf("-Recall:     %.3f (Sensitivity)\n", eval.recall(1));
            System.out.printf("-F1-Score:   %.3f\n", eval.fMeasure(1));
            System.out.printf("-AUC:        %.3f\n", eval.areaUnderROC(1));
        } catch (Exception e) {
            System.out.println("Class-specific metrics not available");
        }

        System.out.printf("\nRuntime:    %.2f seconds\n", runtime);

        System.out.println("\n--- Confusion Matrix ---");
        try {
            System.out.println(eval.toMatrixString());
        } catch (Exception e) {
            e.printStackTrace();
        }
        //Additional statistics
        System.out.println("\nAdditional Statistics:");
        System.out.printf("-Kappa statistic:      %.3f\n", eval.kappa());
        System.out.printf("-Mean absolute error:  %.4f\n", eval.meanAbsoluteError());
        System.out.printf("-Root mean squared error: %.4f\n", eval.rootMeanSquaredError());
    }

    private void saveEvaluationReport(String modelName, Evaluation eval,
                                      double runtime, String reportPath) throws IOException {
        // Create output directory if not exists
        File reportFile = new File(reportPath);
        File parentDir = reportFile.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }
        //Append to report file
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(reportPath, true))) {

            writer.write("=".repeat(70) + "\n");
            writer.write("Model: " + modelName + "\n");
            writer.write("=".repeat(70) + "\n\n");
            //Overall metrics
            writer.write("Overall Performance:\n");
            writer.write(String.format("-Accuracy:           %.2f%%\n", eval.pctCorrect()));
            writer.write(String.format("-Precision (Weighted): %.3f\n", eval.weightedPrecision()));
            writer.write(String.format("-Recall (Weighted):    %.3f\n", eval.weightedRecall()));
            writer.write(String.format("-F1-Score (Weighted):  %.3f\n", eval.weightedFMeasure()));
            writer.write(String.format("-Runtime:             %.2f seconds\n\n", runtime));
            //Disease class metrics
            try {
                writer.write("Disease Class Metrics (Class 1):\n");
                writer.write(String.format("-Precision:  %.3f\n", eval.precision(1)));
                writer.write(String.format("-Recall:     %.3f (Sensitivity)\n", eval.recall(1)));
                writer.write(String.format("-F1-Score:   %.3f\n", eval.fMeasure(1)));
                writer.write(String.format("-AUC:        %.3f\n\n", eval.areaUnderROC(1)));
            } catch (Exception e) {
                writer.write("Class-specific metrics not available\n\n");
            }
            //Confusion Matrix
            try {
                writer.write(eval.toMatrixString() + "\n");
            } catch (IOException e) {
                e.printStackTrace();
            } catch (Exception e) {
                e.printStackTrace();
            }
            // Detailed statistics
            try {
                writer.write(eval.toClassDetailsString() + "\n");
            } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            } catch (Exception e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
            // Additional metrics
            writer.write("Additional Metrics:\n");
            writer.write(String.format("-Kappa statistic:      %.3f\n", eval.kappa()));
            writer.write(String.format("-Mean absolute error:  %.4f\n", eval.meanAbsoluteError()));
            writer.write(String.format("-RMSE:                 %.4f\n", eval.rootMeanSquaredError()));
            writer.write("\n" + "=".repeat(70) + "\n\n");
        }
        System.out.println("\nReport appended to: " + reportPath);
    }
    /**
     Compare all evaluated models
     @param reportPath Path to save comparison report
     @throws Exception
     */
    public void compareModels(String reportPath) throws Exception {
        if (results.isEmpty()) {
            System.out.println("No models to compare!");
            return;
        }
        System.out.println("\n" + "=".repeat(60));
        System.out.println("MODEL COMPARISON");
        System.out.println("=".repeat(60));
        // Print comparison table to console
        printComparisonTable();
        // Print winner analysis
        analyzeWinner();
        //Save detailed comparison to file
        saveComparisonReport(reportPath);
        //ROC comparison
        compareROC();
        System.out.println("\nModel comparison completed!");
    }

    /**
     * Print comparison table
     */
    private void printComparisonTable() {
        System.out.println("\n┌─────────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐");
        System.out.println("│ Model               │ Accuracy │ Precision│  Recall  │ F1-Score │   AUC    │");
        System.out.println("├─────────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤");

        for (EvaluationResult result : results) {
            try {
                System.out.printf("│ %-19s │  %6.2f%% │   %.3f  │   %.3f  │   %.3f  │   %.3f  │\n",
                        truncate(result.modelName, 19),
                        result.evaluation.pctCorrect(),
                        result.evaluation.precision(1),
                        result.evaluation.recall(1),
                        result.evaluation.fMeasure(1),
                        result.evaluation.areaUnderROC(1));
            } catch (Exception e) {
                System.out.printf("│ %-19s │  %6.2f%% │   N/A   │   N/A   │   N/A   │   N/A   │\n",
                        truncate(result.modelName, 19),
                        result.evaluation.pctCorrect());
            }
        }
        System.out.println("└─────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘");
    }

    private void analyzeWinner() {
        System.out.println("\nWinner Analysis");

        // Find best by different metrics
        EvaluationResult bestAccuracy = findBest("accuracy");
        EvaluationResult bestRecall = findBest("recall");
        EvaluationResult bestF1 = findBest("f1");
        EvaluationResult bestAUC = findBest("auc");

        System.out.println("\nBest Models:");
        System.out.println("-Accuracy:  " + bestAccuracy.modelName +
                " (" + String.format("%.2f%%", bestAccuracy.evaluation.pctCorrect()) + ")");

        try {
            System.out.println("-Recall:    " + bestRecall.modelName +
                    " (" + String.format("%.3f", bestRecall.evaluation.recall(1)) + ")");
            System.out.println("-F1-Score:  " + bestF1.modelName +
                    " (" + String.format("%.3f", bestF1.evaluation.fMeasure(1)) + ")");
            System.out.println("-AUC:       " + bestAUC.modelName +
                    " (" + String.format("%.3f", bestAUC.evaluation.areaUnderROC(1)) + ")");
        } catch (Exception e) {
            System.out.println("(Some metrics not available)");
        }

        // Overall winner (most wins)
        String overallWinner = determineOverallWinner(bestAccuracy, bestRecall, bestF1, bestAUC);
        System.out.println("\nOverall Winner: " + overallWinner);
    }

    /**
     * Find best model by metric
     */
    private EvaluationResult findBest(String metric) {
        EvaluationResult best = results.get(0);

        for (EvaluationResult result : results) {
            try {
                switch (metric.toLowerCase()) {
                    case "accuracy":
                        if (result.evaluation.pctCorrect() > best.evaluation.pctCorrect()) {
                            best = result;
                        }
                        break;
                    case "recall":
                        if (result.evaluation.recall(1) > best.evaluation.recall(1)) {
                            best = result;
                        }
                        break;
                    case "f1":
                        if (result.evaluation.fMeasure(1) > best.evaluation.fMeasure(1)) {
                            best = result;
                        }
                        break;
                    case "auc":
                        if (result.evaluation.areaUnderROC(1) > best.evaluation.areaUnderROC(1)) {
                            best = result;
                        }
                        break;
                }
            } catch (Exception e) {
                // Skip if metric not available
            }
        }

        return best;
    }

    /**
     * Determine overall winner
     */
    private String determineOverallWinner(EvaluationResult... bests) {
        // Count wins for each model
        java.util.Map<String, Integer> wins = new java.util.HashMap<>();

        for (EvaluationResult best : bests) {
            wins.put(best.modelName, wins.getOrDefault(best.modelName, 0) + 1);
        }

        // Find model with most wins
        String winner = "";
        int maxWins = 0;

        for (java.util.Map.Entry<String, Integer> entry : wins.entrySet()) {
            if (entry.getValue() > maxWins) {
                maxWins = entry.getValue();
                winner = entry.getKey();
            }
        }

        return winner + " (" + maxWins + "/4 metrics)";
    }

    /**
     * Save comparison report to file
     */
    private void saveComparisonReport(String reportPath) throws IOException {
        File reportFile = new File(reportPath);

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(reportFile, true))) {

            writer.write("\n\n");
            writer.write("=".repeat(70) + "\n");
            writer.write("MODEL COMPARISON SUMMARY\n");
            writer.write("=".repeat(70) + "\n\n");

            // Comparison table
            writer.write("Performance Comparison:\n");
            writer.write("-".repeat(70) + "\n");
            writer.write(String.format("%-20s %10s %10s %10s %10s\n",
                    "Model", "Accuracy", "Recall", "F1-Score", "AUC"));
            writer.write("-".repeat(70) + "\n");

            for (EvaluationResult result : results) {
                try {
                    writer.write(String.format("%-20s %9.2f%% %10.3f %10.3f %10.3f\n",
                            result.modelName,
                            result.evaluation.pctCorrect(),
                            result.evaluation.recall(1),
                            result.evaluation.fMeasure(1),
                            result.evaluation.areaUnderROC(1)));
                } catch (Exception e) {
                    writer.write(String.format("%-20s %9.2f%% %10s %10s %10s\n",
                            result.modelName,
                            result.evaluation.pctCorrect(),
                            "N/A", "N/A", "N/A"));
                }
            }

            writer.write("\n" + "=".repeat(70) + "\n");
        }
    }

    /**
     * Compare ROC curves
     */
    private void compareROC() {
        System.out.println("\nROC Curve Comparison");

        for (EvaluationResult result : results) {
            try {
                ThresholdCurve tc = new ThresholdCurve();
                Instances rocData = tc.getCurve(result.evaluation.predictions(), 1);
                double auc = ThresholdCurve.getROCArea(rocData);

                System.out.printf("%-20s: AUC = %.3f", result.modelName, auc);

                // Interpretation
                if (auc >= 0.9) {
                    System.out.println(" (Excellent)");
                } else if (auc >= 0.8) {
                    System.out.println(" (Good)");
                } else if (auc >= 0.7) {
                    System.out.println(" (Fair)");
                } else {
                    System.out.println(" (Poor)");
                }

            } catch (Exception e) {
                System.out.println(result.modelName + ": ROC not available");
            }
        }
    }

    /**
     * Compare J48 Before vs After Preprocessing
     * Creates a dedicated comparison table for J48 performance improvement
     */
    public void compareJ48BeforeAfterPreprocessingComparision(String reportPath) throws IOException {
        if (results.size() < 2) {
            System.out.println("\nNeed at least 2 J48 evaluations for before/after comparison!");
            return;
        }

        // Assume first two results are J48 (raw) and J48 (preprocessed)
        EvaluationResult j48Raw = results.get(0);
        EvaluationResult j48Prep = results.get(1);

        System.out.println("\n" + "=".repeat(60));
        System.out.println("J48: BEFORE vs AFTER PREPROCESSING");
        System.out.println("=".repeat(60));

        System.out.println("\n┌─────────────────────┬───────────────┬──────────────┐");
        System.out.println("│ Metric              │ Non-preprocess│ Preprocessed │");
        System.out.println("├─────────────────────┼───────────────┼──────────────┤");

        try {
            System.out.printf("│ Accuracy            │   %6.2f%%     │   %6.2f%%    │\n",
                    j48Raw.evaluation.pctCorrect(),
                    j48Prep.evaluation.pctCorrect());
            System.out.printf("│ Precision           │    %.3f      │    %.3f     │\n",
                    j48Raw.evaluation.precision(1),
                    j48Prep.evaluation.precision(1));
            System.out.printf("│ Recall              │    %.3f      │    %.3f     │\n",
                    j48Raw.evaluation.recall(1),
                    j48Prep.evaluation.recall(1));
            System.out.printf("│ F1-Score            │    %.3f      │    %.3f     │\n",
                    j48Raw.evaluation.fMeasure(1),
                    j48Prep.evaluation.fMeasure(1));
            System.out.printf("│ Runtime             │  %.2f sec     │  %.2f sec    │\n",
                    j48Raw.runtime,
                    j48Prep.runtime);

            System.out.println("└─────────────────────┴───────────────┴──────────────┘");


        } catch (Exception e) {
            System.out.println("Error generating comparison: " + e.getMessage());
        }

        // Save to report file
        saveJ48ComparisonReport(j48Raw, j48Prep, reportPath);
    }

    /**
     * Save J48 before/after comparison to report file
     */
    private void saveJ48ComparisonReport(EvaluationResult j48Raw, EvaluationResult j48Prep,
                                         String reportPath) throws IOException {
        File reportFile = new File(reportPath);

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(reportFile, true))) {
            writer.write("\n\n");
            writer.write("=".repeat(70) + "\n");
            writer.write("J48 before and after preprocessing\n");
            writer.write("=".repeat(70) + "\n\n");

            writer.write(String.format("%-20s %15s %15s\n", "Metric", "Non-preprocess", "Preprocessed"));
            writer.write("-".repeat(70) + "\n");

            try {
                writer.write(String.format("%-20s %14.2f%% %14.2f%%\n", "Accuracy",
                        j48Raw.evaluation.pctCorrect(),
                        j48Prep.evaluation.pctCorrect()));
                writer.write(String.format("%-20s %15.3f %15.3f\n", "Precision",
                        j48Raw.evaluation.precision(1),
                        j48Prep.evaluation.precision(1)));
                writer.write(String.format("%-20s %15.3f %15.3f\n", "Recall",
                        j48Raw.evaluation.recall(1),
                        j48Prep.evaluation.recall(1)));
                writer.write(String.format("%-20s %15.3f %15.3f\n", "F1-Score",
                        j48Raw.evaluation.fMeasure(1),
                        j48Prep.evaluation.fMeasure(1)));
                writer.write(String.format("%-20s %12.2f sec %12.2f sec\n", "Runtime",
                        j48Raw.runtime,
                        j48Prep.runtime));

            } catch (Exception e) {
                writer.write("Error writing comparison details\n");
            }

            writer.write("\n" + "=".repeat(70) + "\n");
        }

        System.out.println("J48 comparison report saved to: " + reportPath);
    }

    public void compareRFBeforeAfterPreprocessingComparision(String reportPath) throws IOException {
        if (results.size() < 4) {
            System.out.println("\nNeed at least 4 evaluations (J48 raw, RF raw, J48 prep, RF improved)!");
            return;
        }

        // Lấy đúng theo thứ tự trong pipeline của mày
        EvaluationResult rfRaw = results.get(1);      // RandomForest trên dữ liệu thô
        EvaluationResult rfImproved = results.get(3); // RandomForest sau SMOTE + Feature Selection

        System.out.println("\n" + "=".repeat(60));
        System.out.println("RANDOM FOREST: BEFORE vs AFTER PREPROCESSING + SMOTE + FEATURE SELECTION");
        System.out.println("=".repeat(60));

        System.out.println("\n┌─────────────────────┬───────────────┬──────────────┐");
        System.out.println("│ Metric              │   Raw Data    │   Improved    │");
        System.out.println("├─────────────────────┼───────────────┼──────────────┤");

        try {
            System.out.printf("│ Accuracy            │   %7.2f%%    │   %7.2f%%    │\n",
                    rfRaw.evaluation.pctCorrect(),
                    rfImproved.evaluation.pctCorrect());
            System.out.printf("│ Precision (Class 1) │     %.3f     │     %.3f     │\n",
                    rfRaw.evaluation.precision(1),
                    rfImproved.evaluation.precision(1));
            System.out.printf("│ Recall (Class 1)    │     %.3f     │     %.3f     │\n",
                    rfRaw.evaluation.recall(1),
                    rfImproved.evaluation.recall(1));
            System.out.printf("│ F1-Score (Class 1)  │     %.3f     │     %.3f     │\n",
                    rfRaw.evaluation.fMeasure(1),
                    rfImproved.evaluation.fMeasure(1));
            System.out.printf("│ AUC (Class 1)       │     %.3f     │     %.3f     │\n",
                    rfRaw.evaluation.areaUnderROC(1),
                    rfImproved.evaluation.areaUnderROC(1));
            System.out.printf("│ Runtime             │  %7.2f sec  │  %7.2f sec  │\n",
                    rfRaw.runtime,
                    rfImproved.runtime);

            System.out.println("└─────────────────────┴───────────────┴──────────────┘");

        } catch (Exception e) {
            System.out.println("Error generating RF comparison: " + e.getMessage());
        }

        // Ghi vào file
        saveRFComparisonReport(rfRaw, rfImproved, reportPath);
    }

    /**
     * Ghi báo cáo so sánh Random Forest vào file (giống hệt J48)
     */
    private void saveRFComparisonReport(EvaluationResult rfRaw, EvaluationResult rfImproved,
                                        String reportPath) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(reportPath, true))) {
            writer.write("\n\n");
            writer.write("=".repeat(70) + "\n");
            writer.write("RANDOM FOREST: BEFORE vs AFTER PREPROCESSING + SMOTE + FEATURE SELECTION\n");
            writer.write("=".repeat(70) + "\n\n");

            writer.write(String.format("%-20s %15s %15s\n", "Metric", "Raw Data", "Improved"));
            writer.write("-".repeat(70) + "\n");

            writer.write(String.format("%-20s %14.2f%% %14.2f%%\n", "Accuracy",
                    rfRaw.evaluation.pctCorrect(), rfImproved.evaluation.pctCorrect()));
            writer.write(String.format("%-20s %15.3f %15.3f\n", "Precision",
                    rfRaw.evaluation.precision(1), rfImproved.evaluation.precision(1)));
            writer.write(String.format("%-20s %15.3f %15.3f\n", "Recall",
                    rfRaw.evaluation.recall(1), rfImproved.evaluation.recall(1)));
            writer.write(String.format("%-20s %15.3f %15.3f\n", "F1-Score",
                    rfRaw.evaluation.fMeasure(1), rfImproved.evaluation.fMeasure(1)));
            writer.write(String.format("%-20s %15.3f %15.3f\n", "AUC",
                    rfRaw.evaluation.areaUnderROC(1), rfImproved.evaluation.areaUnderROC(1)));
            writer.write(String.format("%-20s %12.2f sec %12.2f sec\n", "Runtime",
                    rfRaw.runtime, rfImproved.runtime));

            writer.write("\n" + "=".repeat(70) + "\n");
        }

        System.out.println("Random Forest comparison report saved to: " + reportPath);
    }

    public void showROCCurve(String modelName, Evaluation eval) throws Exception {
        ThresholdCurve tc = new ThresholdCurve();
        Instances curve = tc.getCurve(eval.predictions(), 1); // class 1

        ThresholdVisualizePanel vmc = new ThresholdVisualizePanel();
        vmc.setROCString("(Area Under ROC = " +
                Utils.doubleToString(tc.getROCArea(curve), 4) + ")");
        vmc.setName(modelName + " ROC Curve");

        PlotData2D tempd = new PlotData2D(curve);
        tempd.setPlotName(modelName);
        tempd.addInstanceNumberAttribute();

        vmc.addPlot(tempd);

        JFrame jf = new JFrame("ROC Curve - " + modelName);
        jf.setSize(800, 600);
        jf.getContentPane().add(vmc);
        jf.setVisible(true);
    }


    /**
     * Helper: Truncate string
     */
    private String truncate(String str, int maxLength) {
        if (str.length() <= maxLength) {
            return str;
        }
        return str.substring(0, maxLength - 3) + "...";
    }

    /**
     * Clear stored results (for new pipeline)
     */
    public void clearResults() {
        results.clear();
    }

    /**
     * Get number of evaluated models
     */
    public int getNumberOfModels() {
        return results.size();
    }
}