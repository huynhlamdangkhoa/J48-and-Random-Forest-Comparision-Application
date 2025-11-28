package com.example.algorithms;

import com.example.utils.Helpers;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class RandomForestClassifier implements Algorithm {
    private RandomForest randomForest;

    @Override
    public Instances specificPreprocess(Instances data) throws Exception {
        // Implement specific preprocessing logic if needed
        return data;
    }

    @Override
    public void train(Instances data) throws Exception {
        randomForest = new RandomForest();

        Instances processedData = applyTextProcessing(data);

        Instances balancedData = applySMOTE(processedData);

        Instances filteredData = applyFeatureSelection(balancedData);
        Helpers helper = new Helpers(); 
        helper.exportToCSV(filteredData, "RandomForest_filtered_dataset.csv"); 
        randomForest.buildClassifier(filteredData);
    }

    @Override
    public String getResults() {
        return randomForest.toString();
    }

    @Override
    public Classifier getClassifier() {
        return randomForest;
    }

    @Override
    public Instances applyFeatureSelection(Instances data) throws Exception {
        AttributeSelection selector = new AttributeSelection();
        CfsSubsetEval evaluator = new CfsSubsetEval();
        BestFirst search = new BestFirst();

        selector.setEvaluator(evaluator);
        selector.setSearch(search);
        selector.SelectAttributes(data);
        Instances reducedData = selector.reduceDimensionality(data);
        
        return reducedData;
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
