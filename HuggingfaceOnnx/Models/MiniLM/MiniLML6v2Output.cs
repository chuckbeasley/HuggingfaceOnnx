using Microsoft.ML.Data;

namespace HuggingfaceOnnx.Models.MiniLM
{
    internal class MiniLML6v2Output
    {
        // Dimensions: batch, sequence, hidden_size
        [VectorType(1, 256, 384)]
        [ColumnName("last_hidden_state")]
        public float[] Embedding { get; set; }
    }
}