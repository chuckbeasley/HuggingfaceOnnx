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

        public float[] GenerateEmbeddings(IEnumerable<string> input, bool meanPooling = true, bool normalize = true)
        {
            var modelPath = "Resources/Model/all-MiniLM-L6-v2_quantized.onnx";
            var mlContext = new MLContext();

            var inputTexts = input.ToList();
            var batchSize = inputTexts.Count;
            var encodedCorpus = PrepareInput(inputTexts);

            // Onnx models do not support variable dimension vectors. We're using
            // schema definitions to predict a batch.
            // Input schema dimensions: batchSize x sequence
            var inputSchema = SchemaDefinition.Create(typeof(MiniLML6v2Input));
            inputSchema["input_ids"].ColumnType =
                new VectorDataViewType(
                    NumberDataViewType.Int64,
                    batchSize,
                    encodedCorpus.InputIds.Length);
            inputSchema["attention_mask"].ColumnType =
                new VectorDataViewType(
                    NumberDataViewType.Int64,
                    batchSize,
                    encodedCorpus.AttentionMask.Length);
            inputSchema["token_type_ids"].ColumnType =
                new VectorDataViewType(
                    NumberDataViewType.Int64,
                    batchSize,
                    encodedCorpus.TokenTypeIds.Length);

            // Onnx models may have hardcoded dimensions for inputs. Use a custom
            // schema for variable dimension since the number of text documents
            // are a user input for us (batchSize).
            var inputShape = new Dictionary<string, int[]>
            {
                { "input_ids", new[] { batchSize, encodedCorpus.InputIds.Length } },
                { "attention_mask", new[] { batchSize, encodedCorpus.AttentionMask.Length } },
                { "token_type_ids", new[] { batchSize, encodedCorpus.TokenTypeIds.Length } }
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
                    encodedCorpus.InputIds.Length, //config.MaxSequenceLength,
                    384);

            
            var engine = mlContext.Model
                .CreatePredictionEngine<MiniLML6v2Input, MiniLML6v2Output>(
                    model,
                    inputSchemaDefinition: inputSchema,
                    outputSchemaDefinition: outputSchema);
            var predict = engine.Predict(encodedCorpus);

            if (meanPooling == false && normalize == false)
            {
                return predict.Embedding;
            }
            else if (meanPooling == true && normalize == false)
            {
                return Pooling.MeanPooling(
                  predict.Embedding,
                  encodedCorpus.AttentionMask,
                  batchSize,
                  encodedCorpus.AttentionMask.Length).Data<float>().ToArray();
            }
            else if (meanPooling == true &&  normalize == true)
            {
                return Normalization.Normalize(Pooling.MeanPooling(
                  predict.Embedding,
                  encodedCorpus.AttentionMask,
                  batchSize,
                  encodedCorpus.AttentionMask.Length));
            }
            else // meanPooling == false && normalize == true
            {
                TorchTensor tokenEmbeddings = Float32Tensor.from(
                    predict.Embedding,
                    [batchSize, encodedCorpus.AttentionMask.Length, 384]);
                return Normalization.Normalize(tokenEmbeddings);
            }
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
            var tokens = tokenizer.Tokenize(inputTexts);
            var miniLML6v2Inputs = inputTexts.Select(text => Encode(
                tokens,
                tokens.Count)).ToList();

            // Convert encoded inputs to tensors
            var inputIdsTensor = new DenseTensor<long>(
                miniLML6v2Inputs.SelectMany(b => b.InputIds).ToArray(),
                new[]
                {
                    batchSize, tokens.Count
                });
            var attentionMaskTensor = new DenseTensor<long>(
                miniLML6v2Inputs.SelectMany(b => b.AttentionMask).ToArray(),
                new[]
                {
                    batchSize, tokens.Count
                });
            var tokenTypeIdsTensor = new DenseTensor<long>(
                miniLML6v2Inputs.SelectMany(b => b.TokenTypeIds).ToArray(),
                new[]
                {
                    batchSize, tokens.Count
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