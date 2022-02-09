using Microsoft.ML;

//REF https: //docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/sentiment-analysis

namespace Teximal
{
  /// <summary>
  ///   Class AnalyseSentiment.
  /// </summary>
  public static class AnalyseSentiment
  {
    /// <summary>
    ///   Runs the specified data path.
    /// </summary>
    /// <param name="dataPath">The data path.</param>
    public static void Run(string dataPath)
    {
      var mlContext     = new MLContext();
      var splitDataView = LoadData(mlContext, dataPath);

      var model = BuildAndTrainModel(mlContext, splitDataView.TrainSet);

      Evaluate(mlContext, model, splitDataView.TestSet);

      UseModelWithSingleItem(mlContext, model);
      UseModelWithBatchItems(mlContext, model);
    }

    /// <summary>
    ///   Builds and trains the model.
    /// </summary>
    /// <param name="mlContext">The ml context.</param>
    /// <param name="trainingSet">The training set.</param>
    /// <returns>ITransformer.</returns>
    private static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainingSet)
    {
      var estimator = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.SentimentText))
                               .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression());
      Console.WriteLine("=============== BUILDING AND TRAINING MODEL ===============");
      var model = estimator.Fit(trainingSet);
      Console.WriteLine("=============== DONE TRAINING ===============");
      return model;
    }

    /// <summary>
    ///   Evaluates the specified ml context.
    /// </summary>
    /// <param name="mlContext">The ml context.</param>
    /// <param name="model">The model.</param>
    /// <param name="dataView">The data view.</param>
    private static void Evaluate(MLContext mlContext, ITransformer model, IDataView dataView)
    {
      Console.WriteLine("=============== Evaluating Model accuracy with Test data ===============");
      var predictions = model.Transform(dataView);
      var metrics     = mlContext.BinaryClassification.Evaluate(predictions);

      Console.WriteLine();
      Console.WriteLine("Model quality metrics evaluation");
      Console.WriteLine("--------------------------------");
      Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
      Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
      Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
      Console.WriteLine("=============== End of model evaluation ===============");
    }

    /// <summary>
    ///   Uses the model with single item.
    /// </summary>
    /// <param name="mlContext">The ml context.</param>
    /// <param name="model">The model.</param>
    private static void UseModelWithSingleItem(MLContext mlContext, ITransformer model)
    {
      var predictionFunc = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

      var sampleStatement = new SentimentData {SentimentText = "This was a very bad steak"};

      var result = predictionFunc.Predict(sampleStatement);

      Console.WriteLine();
      Console.WriteLine("=============== Prediction Test of model with a single sample and test dataset ===============");

      Console.WriteLine();
      Console.WriteLine($"Sentiment: {result.SentimentText} | Prediction: {(Convert.ToBoolean(result.Prediction) ? "Positive" : "Negative")} | Probability: {result.Probability} ");

      Console.WriteLine("=============== End of Predictions ===============");
      Console.WriteLine();
    }

    /// <summary>
    ///   Uses the model with batch items.
    /// </summary>
    /// <param name="mlContext">The ml context.</param>
    /// <param name="model">The model.</param>
    private static void UseModelWithBatchItems(MLContext mlContext, ITransformer model)
    {
      IEnumerable<SentimentData> sentiments = new[]
      {
        new SentimentData {SentimentText = "This was a horrible meal"},
        new SentimentData {SentimentText = "This was an okay experience"},
        new SentimentData {SentimentText = "I love this spaghetti."}
      };

      var batchComments = mlContext.Data.LoadFromEnumerable(sentiments);

      var predictions = model.Transform(batchComments);

      // Use model to predict whether comment data is Positive (1) or Negative (0).
      var predictedResults = mlContext.Data.CreateEnumerable<SentimentPrediction>(predictions, false);

      Console.WriteLine();
      Console.WriteLine("=============== Prediction Test of model with a batch samples and test dataset ===============");

      Console.WriteLine();
      foreach (var prediction in predictedResults) Console.WriteLine($"Sentiment: {prediction.SentimentText} | Prediction: {(Convert.ToBoolean(prediction.Prediction) ? "Positive" : "Negative")} | Probability: {prediction.Probability} ");

      Console.WriteLine("=============== End of Predictions ===============");
      Console.WriteLine();
    }

    /// <summary>
    ///   Loads the data.
    /// </summary>
    /// <param name="mlContext">The ml context.</param>
    /// <param name="trainerDataPath">The trainer data path.</param>
    /// <returns>DataOperationsCatalog.TrainTestData.</returns>
    private static DataOperationsCatalog.TrainTestData LoadData(MLContext mlContext, string trainerDataPath)
    {
      var dataView = mlContext.Data.LoadFromTextFile<SentimentData>(trainerDataPath, hasHeader: false);
      return mlContext.Data.TrainTestSplit(dataView, 0.2);
    }
  }
}
