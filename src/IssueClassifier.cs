using Microsoft.ML;
using Microsoft.ML.Data;

namespace Teximal
{
  public static class IssueClassifier
  {
    private static MLContext _mlContext;
    private static PredictionEngine<GitHubIssue, IssuePrediction> _predEngine;
    private static ITransformer _trainedModel;
    private static IDataView _trainingDataView;

    public static void Run(string issueTrainData, string issueTestData)
    {
      Console.WriteLine(issueTrainData);
      Console.WriteLine(issueTestData);

      _mlContext = new MLContext(0);

      _trainingDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(issueTrainData, hasHeader: true);
      var pipeline         = ProcessData();
      var trainingPipeline = BuildAndTrainModel(_trainingDataView, pipeline);
      Evaluate(_trainingDataView.Schema, issueTestData);
    }

    private static IEstimator<ITransformer> BuildAndTrainModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
    {
      var trainingPipeline = pipeline.Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
                                     .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
      _trainedModel = trainingPipeline.Fit(trainingDataView);

      _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, IssuePrediction>(_trainedModel);

      var issue = new GitHubIssue
      {
        Title       = "WebSockets communication is slow in my machine",
        Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
      };
      var prediction = _predEngine.Predict(issue);
      Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");

      return trainingPipeline;
    }

    private static IEstimator<ITransformer> ProcessData()
    {
      var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
                               .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                               .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
                               .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                               .AppendCacheCheckpoint(_mlContext);
      return pipeline;
    }

    private static void Evaluate(DataViewSchema trainingDataViewSchema, string testDataPath)
    {
      var testDataView = _mlContext.Data.LoadFromTextFile<GitHubIssue>(testDataPath, hasHeader: true);
      var testMetrics  = _mlContext.MulticlassClassification.Evaluate(_trainedModel.Transform(testDataView));

      Console.WriteLine("*************************************************************************************************************");
      Console.WriteLine("*       Metrics for Multi-class Classification model - Test Data     ");
      Console.WriteLine("*------------------------------------------------------------------------------------------------------------");
      Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
      Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
      Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
      Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
      Console.WriteLine("*************************************************************************************************************");
    }

    private static void PredictIssue()
    {
      var singleIssue = new GitHubIssue {Title = "Entity Framework crashes", Description = "When connecting to the database, EF is crashing"};
      var prediction  = _predEngine.Predict(singleIssue);
      Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
    }
  }

  public class GitHubIssue
  {
    [LoadColumn(0)] public string ID { get; set; }

    [LoadColumn(1)] public string Area { get; set; }

    [LoadColumn(2)] public string Title { get; set; }

    [LoadColumn(3)] public string Description { get; set; }
  }

  public class IssuePrediction
  {
    [ColumnName("PredictedLabel")] public string Area;
  }
}
