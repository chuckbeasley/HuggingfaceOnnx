using HuggingfaceOnnx.Models;
using HuggingfaceOnnx.Models.MiniLM;
using TorchSharp.Tensor;

namespace HuggingfaceOnnx
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var miniLM = new MiniLML6v2(new MiniLML6v2Config());

            string[] query1 = { "That is a happy person" };
            string[] query2 = { "That is a happy person" };
            var query1Embeddings = miniLM.GenerateEmbeddings(query1);
            var query2Embeddings = miniLM.GenerateEmbeddings(query2);
            TorchTensor corpus = Float32Tensor.from(
                    query1Embeddings,
                    [1, query1Embeddings.Length]);
            TorchTensor query = Float32Tensor.from(
                    query2Embeddings,
                    [1, query2Embeddings.Length]);
            var topK = Similarity.TopKByCosineSimilarity(
                corpus,
                query,
                query1.Length);

            var scores = topK.Values.Data<float>().GetEnumerator();
            foreach (var index in topK.Indexes.Data<long>().ToArray())
            {
                scores.MoveNext();
                Console.WriteLine($"Cosine similarity score: {scores.Current*100:f12}");
                Console.WriteLine();
            }
            
            var dotP = Similarity.DotProduct(query1Embeddings, query2Embeddings);
            Console.WriteLine($"Dot product similarity score: {dotP * 100:f12}");
        }
    }
}