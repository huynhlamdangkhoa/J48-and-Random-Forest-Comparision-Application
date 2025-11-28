package com.example.algorithms;

import java.util.Arrays;
import java.util.Locale;
import java.util.Random;

import com.example.utils.Helpers;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.supervised.instance.SMOTE;

public class J48Classifier implements Algorithm {
    private static final String[] DEFAULT_OPTIONS = {"-C", "0.25", "-M", "2"};
    private static final String[] CLASS_NAME_CANDIDATES = {
        "heart_disease", "heart disease status", "heart_disease_status",
        "target", "num", "diagnosis", "condition", "class"
    };

    private J48 tree;
    private double bestF1Score = 0.0;
    private String[] bestOptions = DEFAULT_OPTIONS.clone();
    private Instances trainingData;

    @Override
    public Instances specificPreprocess(Instances data) throws Exception {
        Instances workingCopy = new Instances(data);
        workingCopy = ensureClassAttribute(workingCopy);
        logClassDistribution("Before SMOTE", workingCopy);
        workingCopy = applySMOTE(workingCopy);
        logClassDistribution("After SMOTE", workingCopy);
        workingCopy = applyDiscretize(workingCopy);
        return applyFeatureSelection(workingCopy);
    }

    @Override
    public void train(Instances data) throws Exception {
        Instances processedData = specificPreprocess(data);
        this.trainingData = new Instances(processedData);
        optimizeJ48(processedData);

        Helpers helper = new Helpers();
        helper.exportToCSV(processedData, "j48_filtered_dataset.csv");

        tree = new J48();
        tree.setOptions(bestOptions);
        tree.buildClassifier(processedData);
    }

    public Instances getTrainingData() {
        return trainingData == null ? null : new Instances(trainingData);
    }

    @Override
    public String getResults() {
        return tree == null ? "" : tree.toString();
    }

    @Override
    public J48 getClassifier() {
        return tree;
    }

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

    private void optimizeJ48(Instances data) throws Exception {
        String[][] paramGrid = {
            {"-C", "0.05", "-M", "2"},
            {"-C", "0.10", "-M", "2"},
            {"-C", "0.15", "-M", "4"},
            {"-C", "0.20", "-M", "4"},
            {"-C", "0.25", "-M", "6"},
            {"-C", "0.30", "-M", "6"},
            {"-C", "0.35", "-M", "8"},
            {"-C", "0.40", "-M", "10"},
            {"-U", "-M", "2"},
            {"-U", "-M", "5"}
        };

        bestF1Score = 0.0;
        bestOptions = DEFAULT_OPTIONS.clone();

        for (String[] options : paramGrid) {
            try {
                J48 tempTree = new J48();
                tempTree.setOptions(options);
                tempTree.buildClassifier(data);

                Evaluation eval = new Evaluation(data);
                eval.crossValidateModel(tempTree, data, 10, new Random(1));
                double f1Score = eval.weightedFMeasure();

                System.out.printf("Options: %s | Accuracy: %.2f%% | Weighted F1: %.3f%n",
                    String.join(" ", options), eval.pctCorrect(), f1Score);

                if (f1Score > bestF1Score) {
                    bestF1Score = f1Score;
                    bestOptions = options.clone();
                }
            } catch (Exception e) {
                System.err.println("Error evaluating options " + String.join(" ", options) + ": " + e.getMessage());
            }
        }

        System.out.printf("Best Options: %s | Best Weighted F1: %.3f%n",
            String.join(" ", bestOptions), bestF1Score);
    }

    private Instances applyDiscretize(Instances data) throws Exception {
        Discretize discretize = new Discretize();
        discretize.setUseBetterEncoding(true);
        discretize.setInputFormat(data);
        return Filter.useFilter(data, discretize);
    }

    private Instances applySMOTE(Instances data) throws Exception {
        Instances copy = new Instances(data);
        copy = ensureClassAttribute(copy);
        SMOTE smote = new SMOTE();
        smote.setPercentage(100);
        smote.setNearestNeighbors(5);
        smote.setInputFormat(copy);
        return Filter.useFilter(copy, smote);
    }

    private Instances ensureClassAttribute(Instances data) {
        if (data.classIndex() != -1) {
            return data;
        }
        Attribute classAttr = findAttribute(data, CLASS_NAME_CANDIDATES);
        if (classAttr != null) {
            data.setClassIndex(classAttr.index());
            return data;
        }
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    private Attribute findAttribute(Instances data, String... candidates) {
        if (candidates == null) {
            return null;
        }
        for (String candidate : candidates) {
            if (candidate == null) {
                continue;
            }
            Attribute direct = data.attribute(candidate);
            if (direct != null) {
                return direct;
            }
        }
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
}
