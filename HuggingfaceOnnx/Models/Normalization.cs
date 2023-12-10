using TorchSharp.Tensor;

namespace HuggingfaceOnnx.Models
{
    public static class Normalization
    {
        public static float[] Normalize(TorchTensor input, float p = 2f, int dim = -1, bool keep = true, float eps = 1e-12f)
        {
            TorchTensor denom;
            if (keep == true)
            {
                denom = input.norm(dim, keep, p).clamp_min(eps).expand([-1, -1]);
            }
            else
            {
                denom = input.norm(dim, keep, p).clamp_min(eps);
            }
            return (input / denom).Data<float>().ToArray();
        }
    }
}