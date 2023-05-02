
import java.io.File;
import weka.core.Instances;
import weka.core.WekaPackageManager;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.SerializationHelper;
import weka.classifiers.trees.RandomForest;

/*
 * Test_RandomForest.java
 * This class is to test a Random Forest model on a dataset. Note that the supplied testing set
 * must have the same attributes as the training set, and compatible with the model.
 * 
 * Command-line parameters:
 * <ul>
 *    <li> filename - the dataset to use</li>
 *    <li> modelname - the model to use</li>
 *    <li> -v - Output predictions for test instances into console </li>
 *    <li> --output - Path to save the labeled dataset (extension: .arff) </li>
 * </ul>
 * 
 * Example: (Open the project) 
 *  cd .\src;
 *  javac .\DataPreprocessing.java .\ArgumentParser.java .\Train_RandomForest.java .\Test_RandomForest.java;
 *  java Test_RandomForest ..\data\HepatitisCdata_v2_processed.arff ..\model\RandomForest_v1.model 
 *      --output='D:\Minh\Books\Program\Major Courses\Introduction to Data Mining\Laboratory\Project (04)\project\model\testdata.arff'
 * 
 */

public class Test_RandomForest {
    public static void main(String[] args) throws Exception {
        WekaPackageManager.loadPackages(false); 
        ArgumentParser parser = new ArgumentParser(args);
        String DATASETPATH = parser.GetIndex().get(0);
        String MODELPATH = parser.GetIndex().get(1);

        // This is to assume that the dataset is UNLABELED
        // https://stackoverflow.com/questions/33556543/how-to-save-model-and-apply-it-on-a-test-dataset-on-java
        // https://waikato.github.io/weka-wiki/use_weka_in_your_java_code/#classifying-instances
        
        // 1. Load dataset
        System.out.println("Loading dataset ...");
        DataSource source = new DataSource(DATASETPATH);
        Instances dataset = source.getDataSet();
        for (int i = 0; i < dataset.numAttributes(); i++) {
            if (dataset.attribute(i).name().equals("Category")) {
                dataset.setClassIndex(i);
                break;
            }
        }

        // 2. Load model
        System.out.println("Loading model ...");
        RandomForest rf = (RandomForest) SerializationHelper.read(MODELPATH);

        // 3. Create a copy & Label the instance
        Instances labeled = new Instances(dataset);
        for (int i = 0; i < dataset.numInstances(); i++) {
            double clsLabel = rf.classifyInstance(dataset.instance(i));
            labeled.instance(i).setClassValue(clsLabel);
            if (parser.GetNamedArguments().get("-v") != null) {
                System.out.println("Instance " + i + " predicted as: " + labeled.instance(i).stringValue(dataset.classAttribute()));
            }
        }
        
        // 4. Save the labeled dataset
        System.out.println("Saving labeled dataset ...");
        if (parser.GetNamedArguments().get("--output") != null) {
            String OUTPUTPATH = parser.GetNamedArguments().get("--output");
            ArffSaver saver = new ArffSaver();
            saver.setInstances(labeled);
            saver.setFile(new File(OUTPUTPATH));
            saver.writeBatch();
        }


    }
}
