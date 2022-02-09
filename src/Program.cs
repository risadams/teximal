namespace Teximal
{
  public static class Program
  {
    public static void Main()
    {
      var dataPath = Path.Combine(Environment.CurrentDirectory, "data", "yelp_labelled.txt");

      AnalyseSentiment.Run(dataPath);
    }
  }
}
