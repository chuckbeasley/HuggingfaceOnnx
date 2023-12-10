using Microsoft.ML.Data;
using Microsoft.ML;
using TorchSharp.Tensor;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace HuggingfaceOnnx.Models.MiniLM
{
    internal class MiniLML6v2
    {
        private static readonly string[] OutputColumnNames =
        {
            "last_hidden_state"
        };

        private static readonly string[] InputColumnNames =
        {
            "input_ids", "attention_mask", "token_type_ids"
        };

        private readonly MiniLML6v2Config config;
        private readonly WordPieceTokenizer tokenizer;

        public MiniLML6v2(MiniLML6v2Config config)
        {
            this.config = config;
            tokenizer = new WordPieceTokenizer(File.ReadAllLines("Resources/vocab.txt").ToList());
        }

        public TorchTensor GenerateVectors(IEnumerable<string> input)
        {
            var modelPath = "Resources/Model/all-MiniLM-L6-v2.onnx";
            var mlContext = new MLContext();

            var inputTexts = input.ToList();
            var batchSize = inputTexts.Count;

            // Onnx models do not support variable dimension vectors. We're using
            // schema definitions to predict a batch.
            // Input schema dimensions: batchSize x sequence
            var inputSchema = SchemaDefinition.Create(typeof(MiniLML6v2Input));
            inputSchema["input_ids"].ColumnType =
                new VectorDataViewType(
                    NumberDataViewType.Int64,
                    batchSize,
                    config.MaxSequenceLength);
            inputSchema["attention_mask"].ColumnType =
                new VectorDataViewType(
                    NumberDataViewType.Int64,
                    batchSize,
                    config.MaxSequenceLength);
            inputSchema["token_type_ids"].ColumnType =
                new VectorDataViewType(
                    NumberDataViewType.Int64,
                    batchSize,
                    config.MaxSequenceLength);

            // Onnx models may have hardcoded dimensions for inputs. Use a custom
            // schema for variable dimension since the number of text documents
            // are a user input for us (batchSize).
            var inputShape = new Dictionary<string, int[]>
            {
                { "input_ids", new[] { batchSize, config.MaxSequenceLength } },
                { "attention_mask", new[] { batchSize, config.MaxSequenceLength } },
                { "token_type_ids", new[] { batchSize, config.MaxSequenceLength } }
            };
            var pipeline = mlContext.Transforms
                .ApplyOnnxModel(
                    OutputColumnNames,
                    InputColumnNames,
                    modelPath,
                    inputShape,
                    null,
                    true);

            // Setup the onnx model
            var trainingData = mlContext.Data.LoadFromEnumerable(new List<MiniLML6v2Input>(), inputSchema);
            var model = pipeline.Fit(trainingData);

            // Output schema dimensions: batchSize x sequence x 384
            var outputSchema = SchemaDefinition.Create(typeof(MiniLML6v2Output));
            outputSchema["last_hidden_state"].ColumnType =
                new VectorDataViewType(
                    NumberDataViewType.Single,
                    batchSize,
                    256,
                    384);

            var encodedCorpus = PrepareInput(inputTexts);
            var engine = mlContext.Model
                .CreatePredictionEngine<MiniLML6v2Input, MiniLML6v2Output>(
                    model,
                    inputSchemaDefinition: inputSchema,
                    outputSchemaDefinition: outputSchema);
            var predict = engine.Predict(encodedCorpus);

            return Pooling.MeanPooling(
                predict.Embedding,
                encodedCorpus.AttentionMask,
                batchSize,
                config.MaxSequenceLength);
        }

        public MiniLML6v2Input PrepareInput(string text)
        {
            return Encode(tokenizer.Tokenize(new[] { text }), config.MaxSequenceLength);
        }

        public MiniLML6v2Input PrepareInput(IEnumerable<string> texts)
        {
            var inputTexts = texts.ToList();
            var batchSize = inputTexts.Count;

            // Encode the inputs with Bert Tokenizer
            var miniLML6v2Inputs = inputTexts.Select(text => Encode(
                tokenizer.Tokenize(new[] { text }),
                config.MaxSequenceLength)).ToList();

            // Convert encoded inputs to tensors
            var inputIdsTensor = new DenseTensor<long>(
                miniLML6v2Inputs.SelectMany(b => b.InputIds).ToArray(),
                new[]
                {
                    batchSize, config.MaxSequenceLength
                });
            var attentionMaskTensor = new DenseTensor<long>(
                miniLML6v2Inputs.SelectMany(b => b.AttentionMask).ToArray(),
                new[]
                {
                    batchSize, config.MaxSequenceLength
                });
            var tokenTypeIdsTensor = new DenseTensor<long>(
                miniLML6v2Inputs.SelectMany(b => b.TokenTypeIds).ToArray(),
                new[]
                {
                    batchSize, config.MaxSequenceLength
                });
            return new MiniLML6v2Input
            {
                InputIds = inputIdsTensor.ToArray(),
                AttentionMask = attentionMaskTensor.ToArray(),
                TokenTypeIds = tokenTypeIdsTensor.ToArray()
            };
        }

        private MiniLML6v2Input Encode(
            List<(string Token, int Index)> tokens,
            int maxSequenceLength)
        {
            var padding = Enumerable
                .Repeat(0L, maxSequenceLength - tokens.Count)
                .ToList();

            var tokenIndexes = tokens
                .Select(token => (long)token.Index)
                .Concat(padding)
                .ToArray();

            var segmentIndexes = this.GetSegmentIndexes(tokens)
                .Concat(padding)
                .ToArray();

            var inputMask =
                tokens.Select(o => 1L)
                    .Concat(padding)
                    .ToArray();

            return new MiniLML6v2Input
            {
                InputIds = tokenIndexes,
                AttentionMask = inputMask,
                TokenTypeIds = segmentIndexes
            };
        }

        private IEnumerable<long> GetSegmentIndexes(
            List<(string Token, int Index)> tokens)
        {
            var segmentIndex = 0;
            var segmentIndexes = new List<long>();

            foreach (var (token, _) in tokens)
            {
                segmentIndexes.Add(segmentIndex);

                if (token == WordPieceTokenizer.DefaultTokens.Separation)
                {
                    segmentIndex++;
                }
            }

            return segmentIndexes;
        }
    }
}