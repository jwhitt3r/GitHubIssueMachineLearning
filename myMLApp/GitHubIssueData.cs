using System.Dynamic;
using Microsoft.ML.Data;
namespace GitHubIssueClassification
{
    public class GitHubIssue
    {
        // Structure of the test data
        [LoadColumn(0)]
        public string ID{get; set;}
        [LoadColumn(1)]
        public string Area { get; set; }
        [LoadColumn(2)]
        public string Title { get; set; }
        [LoadColumn(3)]
        public string Description { get; set; }
}

    public class IssuePrediction
    {
        // What we are trying to predict
        [ColumnName("PredictedLabel")] public string Area;
    }
}