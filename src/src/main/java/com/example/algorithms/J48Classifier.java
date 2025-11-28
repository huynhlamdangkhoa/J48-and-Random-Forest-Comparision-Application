package com.example.algorithms;

import java.util.Random;

import com.example.utils.Helpers;

import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class J48Classifier implements Algorithm {
    private J48 tree;
    private double bestF1Score = 0.0;
    private String[] bestOptions;

    @Override
    public Instances specificPreprocess(Instances data) throws Exception {
        Instances processedData = applyTextProcessing(data);

        Instances balancedData = applySMOTE(processedData);

        Discretize discretize = new Discretize();
        discretize.setUseBetterEncoding(true);
        discretize.setInputFormat(balancedData);
        Instances discretizedData = Filter.useFilter(balancedData, discretize);

        // Step 7: Apply feature selection
        Instances filteredFinalData = applyFeatureSelection(discretizedData);
        return filteredFinalData;
    }

    @Override
    public void train(Instances data) throws Exception {
        optimizeJ48(data);
        Instances processedData = applyTextProcessing(data);

        Instances balancedData = applySMOTE(processedData);

        Discretize discretize = new Discretize();
        discretize.setUseBetterEncoding(true);
        discretize.setInputFormat(balancedData);
        Instances discretizedData = Filter.useFilter(balancedData, discretize);

        // Step 7: Apply feature selection
        Instances filteredFinalData = applyFeatureSelection(discretizedData);  

        Helpers helper = new Helpers(); 
        helper.exportToCSV(filteredFinalData, "j48_filtered_dataset.csv");  
        
        tree = new J48();
        tree.setOptions(new String[]{"-U", "-M", "5"});
        tree.buildClassifier(filteredFinalData);
    }

    @Override
    public String getResults() {
        return tree.toString();
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
        search.setOptions(new String[]{"-D", "1", "-N", "10"}); // Max 10 features
    
        filter.setEvaluator(eval);
        filter.setSearch(search);
        filter.setInputFormat(data);
            
        return Filter.useFilter(data, filter);
    }

    private void optimizeJ48(Instances data) throws Exception {
        String[][] paramGrid = {
            {"-C", "0.3", "-M", "5", "-S"},
            {"-C", "0.5", "-M", "10", "-S"}, // 66.66666666666667%
            {"-C", "0.7", "-M", "15", "-S"},
            {"-C", "0.9", "-M", "20", "-S"}, // 66.66666666666667%
            {"-U", "-M", "5"}, // 66.92657569850552%
            {"-R", "-N", "3", "-M", "10"} // 66.92657569850552%
        };
    
        bestF1Score = 0.0;
        bestOptions = null;
    
        for (String[] options : paramGrid) {
            try {
                J48 tempTree = new J48();
                tempTree.setOptions(options);
                tempTree.buildClassifier(data);
    
                Evaluation eval = new Evaluation(data);
                eval.crossValidateModel(tempTree, data, 10, new Random(1));
                double f1Score = eval.pctCorrect();
    
                System.out.println("Options: " + String.join(" ", options) + " | Accuracy: " + f1Score);
                if (f1Score > bestF1Score) {
                    bestF1Score = f1Score;
                    bestOptions = options.clone();
                }
            } catch (Exception e) {
                // System.err.println("Error evaluating options " + String.join(" ", options) + ": " + e.getMessage());
                continue;
            }
        }
    
        // if (bestOptions == null || bestF1Score == 0.0) {
        //     // System.err.println("Warning: No valid options found; using default J48 options");
        //     bestOptions = new String[]{"-C", "0.25", "-M", "2"};
        //     bestF1Score = 0.0;
        // }
        System.out.println("Best Options: " + String.join(" ", bestOptions) + " | Best Accuracy: " + bestF1Score);
    }

    private Instances applyTextProcessing(Instances data) throws Exception {
        StringToWordVector filter = new StringToWordVector();
        filter.setInputFormat(data);
        filter.setIDFTransform(true);
        filter.setOutputWordCounts(true);
        filter.setWordsToKeep(1000);
        filter.setMinTermFreq(5);
        // String attributes: title(0), currency_buyer(3), tags(17), product_color(18),
        // product_variation_size_id(19), shipping_option_name(21), urgency_text(27),
        // origin_country(28), merchant_title(29), merchant_name(30),
        // merchant_info_subtitle(31), merchant_id(34), product_id(36), theme(37)
        filter.setAttributeIndices("1,4,18,19,20,22,28,29,30,31,32,35,37,38");

        filter.setTokenizer(new weka.core.tokenizers.NGramTokenizer());
        weka.core.tokenizers.NGramTokenizer tokenizer = (weka.core.tokenizers.NGramTokenizer) filter.getTokenizer();
        tokenizer.setNGramMinSize(1);
        tokenizer.setNGramMaxSize(2);

        return Filter.useFilter(data, filter);
    }

    private Instances applySMOTE(Instances data) throws Exception {
        if (data.classIndex() < 0) {
            int classIndex = data.attribute("uses_ad_boosts") != null ? data.attribute("uses_ad_boosts").index() : 5;
            data.setClassIndex(classIndex);
        }

        SMOTE smote = new SMOTE();
        smote.setInputFormat(data);
        smote.setPercentage(100);
        return Filter.useFilter(data, smote);
    }
}