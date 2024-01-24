using CategorizeText;
using Microsoft.ML;
using Microsoft.ML.TorchSharp;
using Microsoft.ML.TorchSharp.NasBert;

// Define an ML Context
MLContext mlContext = new()
{
    GpuDeviceId = 0,
    FallbackToCpu = true
};

// Load the TSV data
Console.WriteLine("Loading data.tsv ...");
var dataView = mlContext.Data.LoadFromTextFile<ModelInput>(
    "data.tsv",
    separatorChar: '\t',
    hasHeader: false
);

var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
var trainData = split.TrainSet;
var testData = split.TestSet;

var pipeline = mlContext.Transforms
                        .Conversion.MapValueToKey(outputColumnName: "Label",
                                                  inputColumnName: "Label")
                                   .Append(mlContext.MulticlassClassification
                                                    .Trainers
                                                    .TextClassification(labelColumnName: "Label",
                                                                        sentence1ColumnName: "Text",
                                                                        architecture: BertArchitecture.Roberta))
                                   .Append(mlContext.Transforms
                                                    .Conversion
                                                    .MapKeyToValue(outputColumnName: "PredictedLabel",
                                                                   inputColumnName: "PredictedLabel"));


Console.WriteLine("Training ...");
var model = pipeline.Fit(trainData);

// Save Pipeline (Optional)
mlContext.Model.Save(model, trainData.Schema, "categorization_pipeline.zip");

// Load Pipeline (Optional)
var loadedModel = mlContext.Model.Load("categorization_pipeline.zip", out DataViewSchema loadedSchema);
var predictions = loadedModel.Transform(trainData);

///
/// Save to ONNX (not yet supported in this format, unless you want to tokenize the words)
/// 
// using (var onnxStream = File.OpenWrite("my_model.onnx"))
// {
//     mlContext.Model.ConvertToOnnx(loadedModel, trainData, onnxStream);
// }
// var estimator = mlContext.Transforms.ApplyOnnxModel("./onnx_model.onnx");

// Evaluate the model's performance against the TEST data set
Console.WriteLine("Evaluating model performance...");

// Tranform
var transformTest = model.Transform(testData);
var metrics = mlContext.MulticlassClassification.Evaluate(transformTest);

// Display Evaluation Metrics
Console.WriteLine($"Micro Accuracy: {metrics.MicroAccuracy}");
Console.WriteLine($"Macro Accuracy: {metrics.MacroAccuracy}");
Console.WriteLine($"Log Loss: {metrics.LogLoss}");
Console.WriteLine();

// Generate Confusion Matrix
Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

// Generate a prediction engine
Console.WriteLine("Creating prediction engine ...");
var engine =
    mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

var input = "There were bats flying in the cave.";

// Get your prediction
var inputData = new ModelInput(input);
var result = engine.Predict(inputData);

// Show category
float bestMatch = result.Score[(uint)result.PredictedLabel];
Console.WriteLine($"Category: {(Category)result.PredictedLabel}, Score: {bestMatch:f2}");
Console.WriteLine();