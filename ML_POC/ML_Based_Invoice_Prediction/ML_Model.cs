using System;
using System.Collections.Generic;
using Microsoft.ML;
using System.Linq;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Api;
using System.IO;

namespace ML_Based_Invoice_Prediction
{

    /// <summary>
    /// User defined Datatype used for Email Input and Output
    /// </summary>
    public class Model_Inputs_Outputs
    {
        //Email Subject text as string
        [Column(ordinal: "0")]
        public string EmailSubject { get; set; }

        //Boolean variable to check if Email is an invoice or not
        [Column(ordinal: "1", name: "Label")]
        public bool IsInvoice { get; set; }
    }

    /// <summary>
    /// User defined Datatype used for Predicting if email subject is invoice of not
    /// </summary>
    public class Model_Predictions
    {
        //Boolean variable to check if Email is an invoice or not
        [Column("2", "PredictedLabel")]
        public bool IsInvoice { get; set; }
    }

    /// <summary>
    /// Class used to perform all Model training and predictions
    /// </summary>
    public class ML_Model
    {
        //List of training data points
        static List<Model_Inputs_Outputs> trainingData = new List<Model_Inputs_Outputs>();

        //List of test data points
        static List<Model_Inputs_Outputs> testData = new List<Model_Inputs_Outputs>();

        /// <summary>
        /// Loads the training data using the sample Email subjects
        /// </summary>
        static void LoadTrainingData()
        {
            //Reading the email sample subjects from txt file to list of strings
            List<string> lines = new List<string>();
            lines = File.ReadAllLines(GlobalVariables.trainingDataFilePath).ToList();

            //Loop to split subject and invoice boolean using | and add to training data set
            foreach (string line in lines)
            {
                bool isItInvoice = false;
                string[] partsOfLine = line.Split('|');
                if (partsOfLine[1].ToLower() == "yes")
                    isItInvoice = true;

                trainingData.Add(new Model_Inputs_Outputs()
                {
                    EmailSubject = partsOfLine[0],
                    IsInvoice = isItInvoice
                });

            }
        }

        /// <summary>
        /// Loads the test data using more sample email subjects
        /// </summary>
        static void LoadTestData()
        {
            //Reading the email sample subjects from txt file to list of strings
            List<string> lines = new List<string>();
            lines = File.ReadAllLines(GlobalVariables.testDataFilePath).ToList();

            //Loop to split subject and invoice boolean using | and add to test data set
            foreach (string line in lines)
            {
                bool isItInvoice = false;
                string[] partsOfLine = line.Split('|');
                if (partsOfLine[1].ToLower() == "yes")
                    isItInvoice = true;

                testData.Add(new Model_Inputs_Outputs()
                {
                    EmailSubject = partsOfLine[0],
                    IsInvoice = isItInvoice
                });

            }
        }

        /// <summary>
        /// Method used for training and executing the ML Model to get prediction of inovice
        /// </summary>
        public void Execute_ML_Model()
        {
            //Loop until user chooses to exit the program
            while (true)
            {
                //Call Methods to load training and test data 
                LoadTrainingData();
                LoadTestData();


                //Use ML Context to implement Model 
                MLContext mlContext = new MLContext();

                //Transform data from list to IDataView
                IDataView trainingDataView = mlContext.CreateStreamingDataView<Model_Inputs_Outputs>(trainingData);

                //Define a pipeline so that model uses featurized input
                var pipeline = mlContext.Transforms.Text.FeaturizeText("EmailSubject", "Features").Append(mlContext.BinaryClassification.Trainers.FastTree(numLeaves: 50, numTrees: 50, minDatapointsInLeaves: 1));

                //Input training data into the model
                var model = pipeline.Fit(trainingDataView);

                //Testing model accuracy by validating using test data set
                IDataView testDataView = mlContext.CreateStreamingDataView<Model_Inputs_Outputs>(testData);
                var predictions = model.Transform(testDataView);
                var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
                Console.BackgroundColor = ConsoleColor.White;
                Console.ForegroundColor = ConsoleColor.Black;
                Console.Title = "Machine Learning POC";
                System.Globalization.NumberFormatInfo nfi = new System.Globalization.CultureInfo("en-US", false).NumberFormat;
                nfi.PercentDecimalDigits = 0;
                Console.WriteLine("Accuracy of the Model = " + metrics.Accuracy.ToString("P", nfi));
                


                //User inputs email subject or Exit to stop run
                Console.WriteLine("Enter an Email Subject or Enter Exit to Terminate Program : ");
                string userInputString = Console.ReadLine();

                //Break loop and stop run if Exit is used
                if (userInputString.ToLower() == "exit")
                    break;

                //Use model to make prediction
                var predictionFunction = model.MakePredictionFunction
                                              <Model_Inputs_Outputs, Model_Predictions>(mlContext);

                Model_Inputs_Outputs inputToModel = new Model_Inputs_Outputs();

                inputToModel.EmailSubject = userInputString;

                var invoicePrediction = predictionFunction.Predict(inputToModel);

                //Display on terminal if email is invoice or not
                if (invoicePrediction.IsInvoice)
                    Console.WriteLine("This is an Invoice");

                else
                    Console.WriteLine("This is NOT an Invoice");

                //Add the user input subject to the training data set
                Console.WriteLine("Was the prediction correct Y/N?");

                string userResponse = Console.ReadLine();
                Console.WriteLine(Environment.NewLine);

                string addNewEmailSubject;

                //Logic to add correct invoice classification for email subject
                if (userResponse.ToLower() == "y" && invoicePrediction.IsInvoice)
                    addNewEmailSubject = userInputString + "|Yes";
                else if (userResponse.ToLower() == "y" && !invoicePrediction.IsInvoice)
                    addNewEmailSubject = userInputString + "|No";
                else if (userResponse.ToLower() != "y" && invoicePrediction.IsInvoice)
                    addNewEmailSubject = userInputString + "|No";
                else
                    addNewEmailSubject = userInputString + "|Yes";

                //Adding subject to training data txt file
                File.AppendAllText(GlobalVariables.trainingDataFilePath, Environment.NewLine + addNewEmailSubject);
            }

            //Completed Loop
            Console.WriteLine("Completed Execution.....");
        }
    }
}
