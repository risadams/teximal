using Microsoft.ML.Data;

namespace Teximal
{
  public class SentimentData
  {
    [LoadColumn(1)] [ColumnName("Label")] public bool Sentiment;

    [LoadColumn(0)] public string? SentimentText;
  }

  public class SentimentPrediction : SentimentData
  {
    [ColumnName("PredictedLabel")] public bool Prediction { get; set; }

    public float Probability { get; set; }
    public float Score { get; set; }
  }
}
