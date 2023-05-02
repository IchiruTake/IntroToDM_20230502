## Getting Started
This program is the the project for the course "Introduction to Data Mining" at International University (02/05/2023)

To run this projram, you must 
1. Install the Weka program
2. Using the Maven dependency manager or any tools such as VSCode to support you. You can also add environment variables CLASSPATH to Weka. The path is C:\Program Files\Weka-3-8-6\weka.jar (version dependent)
3. Open the project and cd to the src: `cd .\src`
4. Compile the program: `javac .\DataPreprocessing.java .\ArgumentParser.java .\Train_RandomForest.java .\Test_RandomForest.java`
5. Enjoy the project. If the absolute path has spacing, add the ' or " to the path.
 - Train: `java Train_RandomForest ..\data\HepatitisCdata_v2_processed.arff  --output=..\model\RandomForest_v1.model`
 - Test: `java Test_RandomForest ..\data\HepatitisCdata_v2_processed.arff ..\model\RandomForest_v1.model  --output="D:\Minh\Books\Program\Major Courses\Introduction to Data Mining\Laboratory\Project (04)\project\model\testdata.arff"`


## Credits
ITITIU19031 – Phạm Hoàng Minh (Leader)
ITITIU19143 – Lê Nguyễn Anh Khoa
ITITIU19033 – Vũ Hoàng Nam
ITITIU18202 – Ksor Lý Tiểu Dũng
