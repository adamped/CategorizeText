using Microsoft.ML.Data;

namespace CategorizeText
{
    public class ModelInput
    {
        public ModelInput(string text)
        {
            Text = text;
        }

        [LoadColumn(0)]
        [ColumnName(@"Text")]
        public string Text { get; set; }

        [LoadColumn(1)]
        [ColumnName(@"Label")]
        public float Label { get; set; }
    }
}
