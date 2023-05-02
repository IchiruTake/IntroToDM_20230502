import java.io.File;
import java.util.HashMap;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.WekaPackageManager;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RenameNominalValues;

/*
 * DataPreprocessing.java
 * This class is to preprocess the dataset before training using the CSV file.
 * Since the original data contained the "NA" to represent missing values, which is not the default format for 
 * missing value of ARFF file (by Weka), we need to preprocess the data. If it is ARFF file, we can use the
 * file directly.
 * 
 * The preprocessing steps are:
 * - (Optional) Configure missing values
 * - Class re-arrangement: Merge 'category' and 'Sex' by RenameNominalValues
 * - Drop 'Unnamed: 0' column to standardize the dataset
 * - Drop columns
 * 
 * Command-line parameters:
 * <ul>
 *    <li> filename - the dataset to use</li>
 *    <li> --drop-useless - Drop columns that have low contribution to the model (Discovered by RandomForest in Python Colab)</li>
 *    <li> --drop-columns - User-defined a list of columns, separated by comma</li>
 *    <li> --output - The output file path. If not specified, change to <filename>_processed.arff</li>
 * </ul>
 * 
 * Example: (Open the project) 
 *  cd .\src;
 *  javac .\DataPreprocessing.java .\ArgumentParser.java .\Train_RandomForest.java .\Test_RandomForest.java;
 *  java DataPreprocessing ..\data\HepatitisCdata.csv --drop-useless --output=..\data\HepatitisCdata_v2_preprocessed.arff
 * 
 */

public class DataPreprocessing {
    public static void main(String[] args) throws Exception {
        WekaPackageManager.loadPackages(false);
        
        ArgumentParser parser = new ArgumentParser(args);
        // ArrayList<String> arguments = parser.GetArguments();
        HashMap<Integer, String> index = parser.GetIndex();
        HashMap<String, String> named_arguments = parser.GetNamedArguments();

        String DATASETPATH = index.get(0);
        
        // 0. Load dataset
        System.out.println("Loading dataset ...");
        Instances dataset;
        if (DATASETPATH.endsWith(".csv")) {
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(DATASETPATH));
            // Set missing value to "NA" using CSVLoader, if not ARFF does not recognize missing values
            loader.setMissingValue("NA");
            dataset = loader.getDataSet();
        } else {
            DataSource source = new DataSource(DATASETPATH);
            dataset = source.getDataSet();
        }

        // 1. Class re-arrangement
        
        // 1.1 In class 'Category', we have '0=Blood Donor', '0s=suspect Blood Donor', '1=Hepatitis' , '2=Fibrosis' , '3=Cirrhosis'
        // We need to merge '0=Blood Donor' and '0s=suspect Blood Donor' into '0'
        // We need to merge '1=Hepatitis' , '2=Fibrosis' , '3=Cirrhosis' into '1'
        
        RenameNominalValues rename = new RenameNominalValues();
        String[] options = new String[2];
        options[0] = "-R";
        options[1] = "Category";
        rename.setOptions(options);
        options[0] = "-N";
        options[1] = "0=Blood Donor:0, 0s=suspect Blood Donor:0, 1=Hepatitis:1, 2=Fibrosis:1, 3=Cirrhosis:1";
        rename.setOptions(options);
        // for (String option : rename.getOptions()) { System.out.println(option); }
        rename.setInputFormat(dataset);
        dataset = Filter.useFilter(dataset, rename);
        // System.out.println(dataset.attribute("Category"));

        // 1.2 In class 'Sex', we have 'f' and 'm'
        // Convert 'm' to '0', and 'f' to '1'
        Attribute sex_attr = dataset.attribute("Sex");
        dataset.renameAttributeValue(sex_attr, "m", "0");
        dataset.renameAttributeValue(sex_attr, "f", "1");

        // 2.0 Remove attributes if found 
        // 2.1 Remove 'Unnamed: 0' attribute if found
        if (dataset.attribute("Unnamed: 0") != null) {
            dataset.deleteAttributeAt(dataset.attribute("Unnamed: 0").index());
        }

        // 3.0 Drop columns if user-defined
        if (named_arguments.get("--drop-columns") != null) {
            System.out.println("Dropping columns ...");
            String[] columns = named_arguments.get("--drop-columns").split(",");
            for (String column : columns) {
                if (dataset.attribute(column) != null) {
                    dataset.deleteAttributeAt(dataset.attribute(column).index());
                }
            }
        }

        // 4.0 Remove records that contains null values
        System.out.println("Removing records with null values ...");
        if (named_arguments.containsKey("--drop-useless")) {
            String[] useless_attr = {"Sex", "Age", "PROT", "CREA", "CHOL"}; // Add "CHE" in the future
            for (String attr : useless_attr) {
                if (dataset.attribute(attr) != null) {
                    dataset.deleteAttributeAt(dataset.attribute(attr).index());
                }
            }
        }
        for (int i = 0; i < dataset.numAttributes(); i++) {
            if (dataset.attribute(i) != null) {
                dataset.deleteWithMissing(dataset.attribute(i));
            }
        }

        // 5.0 Save the preprocessed dataset into a new file that have the .arff extension
        System.out.println("Saving preprocessed dataset ...");
        dataset.compactify();
        String NEW_DATASETPATH = named_arguments.get("--output");
        if (NEW_DATASETPATH == null) {
            NEW_DATASETPATH = DATASETPATH.substring(0, DATASETPATH.lastIndexOf(".")) + "_processed.arff";
        }
        if (!NEW_DATASETPATH.endsWith(".arff")) {
            NEW_DATASETPATH = NEW_DATASETPATH.substring(0, NEW_DATASETPATH.lastIndexOf(".")) + ".arff";
        }
        System.out.println("Saving to " + NEW_DATASETPATH);

        ArffSaver saver = new ArffSaver();
        saver.setInstances(dataset);
        saver.setFile(new File(NEW_DATASETPATH));
        saver.writeBatch();
    }
}
