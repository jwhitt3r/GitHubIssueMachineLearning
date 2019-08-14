using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using GitHubIssueClassification;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace myMLApp
{
    class Program
    {
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _trainDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "issues_train.tsv");
        private static string _myTestDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "myTestData.tsv");
        private static string _testDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "issues_test.tsv");
        private static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Models", "model.zip");

        private static MLContext _mlContext;
        private static PredictionEngine<GitHubIssue, IssuePrediction> _predEngine;
        private static ITransformer _trainedModel;
        private static IDataView _trainingDataView;
        private static IDataView _myTestDataView;
        
        static void Main(string[] args)
        {
            _mlContext = new MLContext(seed: 0); //Random seed, we use 0 to trust that we are repeatable
            _trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_trainDataPath, hasHeader: true); //Load the data into the pipeline
            var pipeline = ProcessData();

            var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline); //Pass in the test data with the class definition, along with the transformed data set
            
            Evaluate(_trainingDataView.Schema); // Evaluate the model
            PredictIssue();

        }

        // Build and Train the Model
        public static IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView,
            IEstimator<ITransformer> pipeline)
        {

            var trainingPipeline = pipeline
                .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            _trainedModel = trainingPipeline.Fit(trainingDataView);
            
            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(_trainedModel);
            GitHubIssue issue = new GitHubIssue()
            {
                Title = "WebSockets communication is slow in my machine",
                Description =
                    "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
            };
            
            // Cross Validation
            /*var cvResults =
                _mlContext.MulticlassClassification.CrossValidate(data: _trainingDataView, trainingPipeline, 5);
            foreach (var r in cvResults)
            {
                Console.WriteLine($"  Fold: {r.Fold}, AUC: {r.Metrics.TopKAccuracy}");
                Console.WriteLine($"  Confusion Matrix: {r.Metrics.ConfusionMatrix}");
            }*/

            Console.WriteLine();

            var prediction = _predEngine.Predict(issue);
            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");
            return trainingPipeline;
        }
        
        //Evaluate the model
        public static void Evaluate(DataViewSchema trainingDataViewSchema)
        {

            var testDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_testDataPath, hasHeader: true);
            var testMetrics = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");
            SaveModelAsFile(_mlContext, trainingDataViewSchema, _trainedModel); // Save our model to file
        }


        public static void PredictIssue()
        {
            
            /*
             * Single Prediction
             * A single issue is created and stored as a new GitHubIssue, which is then used
             * for a prediction via the _predEngine.Predict function.
             */
            
            ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);

            GitHubIssue singleIssue = new GitHubIssue() { Title = "Entity Framework crashes", Description = "When connecting to the database, EF is crashing" }; // Our single issue
            _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(loadedModel);
            var singleprediction = _predEngine.Predict(singleIssue);
            Console.WriteLine($"=============== Single Prediction - Result: {singleprediction.Area} ===============");
            
            
            /*
            * IEnumerable-based Batch Predictions
            * The following block of code, demonstrates the creation of manual entries and storing
             * them into area. Each of the indexes within the array is then predicted via the _predEngine.Predict function
            */
            
            IEnumerable<GitHubIssue> issues = new[]
            {
                new GitHubIssue
                {
                    Title = "Entity Framework crashes",
                    Description = "When connecting to the database, EF is crashing"
                },
                new GitHubIssue
                {
                Title = "Github Down",
                Description = "When going to the website, github says it is down"
                }

            };

            var batchPrediction = _predEngine;


            IDataView batchIssues = _mlContext.Data.LoadFromEnumerable(issues);
            

            IEnumerable<GitHubIssue> predictedResults =
                _mlContext.Data.CreateEnumerable<GitHubIssue>(batchIssues, reuseRowObject: false);

            foreach (GitHubIssue prediction in predictedResults)
            {
                Console.WriteLine($"=========================== Enumerated-based Batch Predictions ==================");
                Console.WriteLine($"*-> Title: {prediction.Title} | Prediction: {batchPrediction.Predict(prediction)}");
                Console.WriteLine($"=================================================================================");
            }


            
            /*
             * File-based Batch Predictions
             * The following block of code, demonstrates the ingestion of a file (myTestData.tsv) which has no
             * Area, to then predictteed via the _predEngine.Predict function.
             */
            _myTestDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(_myTestDataPath, hasHeader: true);

            IEnumerable<GitHubIssue> filePredictedResults =
                _mlContext.Data.CreateEnumerable<GitHubIssue>(_myTestDataView, reuseRowObject: false);
            
            foreach (GitHubIssue prediction in filePredictedResults)
            {
                Console.WriteLine($"============================= File-based Batch Predictions ===========================");
                Console.WriteLine($"*-> Title: {prediction.Title} | Prediction: {batchPrediction.Predict(prediction).Area}");
                Console.WriteLine($"======================================================================================");
            }

        }
        public static void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            mlContext.Model.Save(model, trainingDataViewSchema, _modelPath); // Save the model to the path and model.zip

        }
        
        // Feature Extraction and Transformation of Data
        public static IEstimator<ITransformer> ProcessData()
        {
            var pipeline =
                _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label") // Transform our prediction to a label
                    .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title",
                        outputColumnName: "TitleFeaturized"))
                    .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description",
                        outputColumnName: "DescriptionFeaturized"))
                    .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized")) // Place the title and description under the features column
                    .AppendCacheCheckpoint(_mlContext); // Use cache to iterate over to increase speed, only use for small datasets

            return pipeline;
        }
    }
}