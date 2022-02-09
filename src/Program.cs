namespace Teximal
{
  public static class Program
  {
    public static void Main()
    {
      var sentimentData = Path.Combine(Environment.CurrentDirectory, "data", "yelp_labelled.txt");

      var issueTrainData = Path.Combine(Environment.CurrentDirectory, "data", "issues_train.tsv");
      var issueTestData = Path.Combine(Environment.CurrentDirectory, "data", "issues_test.tsv");

      //AnalyseSentiment.Run(sentimentData);

      IssueClassifier.Run(issueTrainData,issueTestData);
    }
  }
}
