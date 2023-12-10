using HuggingfaceOnnx.Models;
using HuggingfaceOnnx.Models.MiniLM;

namespace HuggingfaceOnnx
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var miniLM = new MiniLML6v2(new MiniLML6v2Config());

            string[] query = { "That is a happy person" };
            string[] query2 = { "That is a happy person" };
            var queryEmbeddings = miniLM.GenerateVectors(query);
            var query2Embeddings = miniLM.GenerateVectors(query2);
            var normalized = Normalization.Normalize(queryEmbeddings, dim: -1, keep: true);
            var normalized2 = Normalization.Normalize(query2Embeddings, dim: -1, keep: true);
            //var normalized = Normalization.Normalize_(queryEmbeddings, dim: 1);
            //var normalized2 = Normalization.Normalize_(query2Embeddings, dim: 1);
            var embeddings = normalized.Data<float>();
            var topK = Similarity.TopKByCosineSimilarity(
                queryEmbeddings,
                query2Embeddings,
                1);

            var scores = topK.Values.Data<float>();
            Console.WriteLine($"Similarity score: {scores[0]:f12}");
        }
    }
}
//denom = input.norm(p, dim, keepdim = True).clamp_min_(eps).expand_as(input)
//        return torch.div(input, denom, out=out)