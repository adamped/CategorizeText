using Microsoft.ML.Data;

namespace CategorizeText
{
    public class ModelOutput
    {
        [ColumnName(@"Text")]
        public string Text { get; set; } = "";

        [ColumnName(@"Label")]
        public uint Label { get; set; }

        [ColumnName(@"PredictedLabel")]
        public float PredictedLabel { get; set; }

        [ColumnName(@"Score")]
        public float[] Score { get; set; } = [];
    }
}
