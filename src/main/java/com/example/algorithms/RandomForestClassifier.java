package com.example.algorithms;

import java.util.Arrays;
import java.util.Locale;
import java.util.Random;

import com.example.utils.Helpers;

import weka.filters.supervised.attribute.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;


public class RandomForestClassifier implements Algorithm {

    private static final String[] CLASS_NAME_CANDIDATES = {
        "heart_disease", "heart disease status", "heart_disease_status",
        "target", "num", "diagnosis", "condition", "class"
    };

    private RandomForest randomForest;
    private Instances trainingData;

    // Tuning
    private double bestF1Score = 0.0;
    private int bestNumTrees = 100;
    private int bestMaxDepth = 0;    // 0 = unlimited
    private int bestNumFeatures = 0; // 0 = default (sqrt)

    @Override
    public void train(Instances data) throws Exception {
        // 1. Pre-proccessing: set class, SMOTE, choosing features
        Instances processedData = specificPreprocess(data);
        this.trainingData = new Instances(processedData);

        // 2. Tuning Random Forest with cross-validation
        optimizeRandomForest(processedData);

        // 3. export dataset after filter report
        Helpers helper = new Helpers();
        helper.exportToCSV(processedData, "RandomForest_filtered_dataset.csv");

        // 4. Build final model with best hyper-parameters 
        randomForest = new RandomForest();
        randomForest.setNumIterations(bestNumTrees);
        randomForest.setSeed(1);
        if (bestMaxDepth > 0) {
            randomForest.setMaxDepth(bestMaxDepth);
        }
        if (bestNumFeatures > 0) {
            randomForest.setNumFeatures(bestNumFeatures);
        }

        randomForest.buildClassifier(processedData);
    }

    @Override
    public String getResults() {
        if (randomForest == null) {
            return "RandomForest has not been trained yet.";
        }

        StringBuilder sb = new StringBuilder();
        sb.append("=== RandomForest Model ===\n");
        sb.append(randomForest.toString());
        sb.append("\n\nBest hyper-parameters (tuned by weighted F1):\n");
        sb.append("  - Trees (numIterations): ").append(bestNumTrees).append('\n');
        sb.append("  - Max depth: ").append(bestMaxDepth == 0 ? "unlimited" : bestMaxDepth).append('\n');
        sb.append("  - Num features per split: ")
        .append(bestNumFeatures == 0 ? "default (sqrt(#features))" : bestNumFeatures)
        .append('\n');
        sb.append(String.format(Locale.ROOT,
                "  - Best weighted F1 (5-fold CV): %.4f%n",
                bestF1Score));

        return sb.toString();
    }

    @Override
    public Classifier getClassifier() {
        return randomForest;
    }

    /**
     * choose attribute with CfsSubsetEval + BestFirst (like J48)
     */
    @Override
    public Instances applyFeatureSelection(Instances data) throws Exception {
        AttributeSelection filter = new AttributeSelection();
        CfsSubsetEval eval = new CfsSubsetEval();
        BestFirst search = new BestFirst();
        search.setOptions(new String[]{"-D", "1", "-N", "10"});

        filter.setEvaluator(eval);
        filter.setSearch(search);
        filter.setInputFormat(data);

        return Filter.useFilter(data, filter);
    }

    /**
     * Pre-proccessing Random Forest:
     * - ensure class attribute is set
     * - apply SMOTE
     * - feature selection
     * @param data Dataset
     * @return Pre-processed dataset
     * @throws Exception
     */
    @Override
    public Instances specificPreprocess(Instances data) throws Exception {
        Instances workingCopy = new Instances(data);

        workingCopy = ensureClassAttribute(workingCopy);
        logClassDistribution("RF - Before SMOTE", workingCopy);

        workingCopy = applySMOTE(workingCopy);
        logClassDistribution("RF - After SMOTE", workingCopy);

        return applyFeatureSelection(workingCopy);
    }

    /**
     * Simple grid search for Random Forest:
     * - numIterations (numTrees): 100, 200, 300
     * - maxDepth: 0 (unlimited), 10, 15
     * - numFeatures: 0 (default), sqrt(#features)
     * Uses 5-fold cross-validation and weighted F1-score for evaluation.
     */
    private void optimizeRandomForest(Instances data) throws Exception {
        int numAttrs = data.numAttributes() - 1; // trừ class
        int defaultNumFeatures = (int) Math.round(Math.sqrt(Math.max(1, numAttrs)));

        int[] numTreesOptions = {100, 200, 300};
        int[] maxDepthOptions = {0, 10, 15};        // 0 = unlimited
        int[] numFeaturesOptions = {0, defaultNumFeatures}; // 0 = default

        bestF1Score = 0.0;
        bestNumTrees = numTreesOptions[0];
        bestMaxDepth = maxDepthOptions[0];
        bestNumFeatures = 0;

        Random rand = new Random(1);

        for (int numTrees : numTreesOptions) {
            for (int maxDepth : maxDepthOptions) {
                for (int numFeatures : numFeaturesOptions) {
                    try {
                        RandomForest candidate = new RandomForest();
                        candidate.setNumIterations(numTrees);
                        candidate.setSeed(1);
                        if (maxDepth > 0) {
                            candidate.setMaxDepth(maxDepth);
                        }
                        if (numFeatures > 0) {
                            candidate.setNumFeatures(numFeatures);
                        }

                        Evaluation eval = new Evaluation(data);
                        eval.crossValidateModel(candidate, data, 5, rand);
                        double f1 = eval.weightedFMeasure();

                        System.out.printf(Locale.ROOT,
                                "RF tuning → trees=%d, maxDepth=%d, numFeatures=%d | Acc=%.2f%% | F1=%.3f%n",
                                numTrees, maxDepth, numFeatures,
                                eval.pctCorrect(), f1);

                        if (f1 > bestF1Score) {
                            bestF1Score = f1;
                            bestNumTrees = numTrees;
                            bestMaxDepth = maxDepth;
                            bestNumFeatures = numFeatures;
                        }
                    } catch (Exception e) {
                        System.err.printf(
                                Locale.ROOT,
                                "Error evaluating RF config (trees=%d, depth=%d, features=%d): %s%n",
                                numTrees, maxDepth, numFeatures, e.getMessage());
                    }
                }
            }
        }

        System.out.printf(Locale.ROOT,
                "RF best config → trees=%d, maxDepth=%d, numFeatures=%d | Best F1=%.3f%n",
                bestNumTrees, bestMaxDepth, bestNumFeatures, bestF1Score);
    }

    /**
     * SMOTE generic, no more hard-code "uses_ad_boosts"
     */
    private Instances applySMOTE(Instances data) throws Exception {
        Instances copy = new Instances(data);
        copy = ensureClassAttribute(copy);

        if (!copy.classAttribute().isNominal()) {
            System.out.println("RandomForest SMOTE skipped: class attribute is not nominal.");
            return copy;
        }

        SMOTE smote = new SMOTE();
        smote.setPercentage(100);
        smote.setNearestNeighbors(5);
        smote.setInputFormat(copy);

        return Filter.useFilter(copy, smote);
    }

    /**
     * Make sure classIndex is set (follow by name, or the last attribute).
     */
    private Instances ensureClassAttribute(Instances data) {
        if (data.classIndex() != -1) {
            return data;
        }

        Attribute classAttr = findAttribute(data, CLASS_NAME_CANDIDATES);
        if (classAttr != null) {
            data.setClassIndex(classAttr.index());
        } else {
            data.setClassIndex(data.numAttributes() - 1);
        }
        return data;
    }

    private Attribute findAttribute(Instances data, String... candidates) {
        if (candidates == null) {
            return null;
        }

        // 1. Try direct match
        for (String candidate : candidates) {
            if (candidate == null) continue;
            Attribute direct = data.attribute(candidate);
            if (direct != null) {
                return direct;
            }
        }

        // 2. Try normalized match
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attr = data.attribute(i);
            if (attr == null) {
                continue;
            }
            String normalized = normalize(attr.name());
            for (String candidate : candidates) {
                if (candidate != null && normalized.equals(normalize(candidate))) {
                    return attr;
                }
            }
        }
        return null;
    }

    private String normalize(String value) {
        return value.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]", "");
    }

    private void logClassDistribution(String stage, Instances data) {
        if (data.classIndex() < 0 || !data.classAttribute().isNominal()) {
            System.out.println(stage + ": class attribute not set or not nominal.");
            return;
        }
        int[] counts = data.attributeStats(data.classIndex()).nominalCounts;
        if (counts == null) {
            System.out.println(stage + ": unable to read class distribution.");
            return;
        }
        System.out.printf("%s → Class counts: %s%n", stage, Arrays.toString(counts));
    }

    // (Optional): get training data for further analysis
    public Instances getTrainingData() {
        return trainingData;
    }
}

