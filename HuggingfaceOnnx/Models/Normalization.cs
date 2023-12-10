using TorchSharp.Tensor;

namespace HuggingfaceOnnx.Models
{
    public static class Normalization
    {
        public static TorchTensor Normalize(TorchTensor input, float p = 2f, int dim = 1, bool keep = false, float eps = 1e-12f)
        {
            var denom = input.norm(dim, keep, p).clamp_min(eps).expand([-1, -1]);
            return input / denom;
        }

        public static TorchTensor Normalize_(TorchTensor input, float p = 2f, int dim = 1)
        {
            dim = SafeIndex(dim, (int)input.Dimensions);

            var norm = input.norm(dim, true, p);

            for (int i = 0; i < input.Data<float>().Length; ++i)
            {
                // Calculate the index in the resulting array
                int resultIndex = 0;
                int resultMultiplier = 1;
                int num = i;

                for (int j = (int)input.Dimensions - 1; j >= 0; --j)
                {
                    int size = (int)input.shape.ElementAt(j);
                    if (j != dim)
                    {
                        int index = num % size;
                        resultIndex += index * resultMultiplier;
                        resultMultiplier *= (int)input.shape.ElementAt(j);
                    }
                    num = (int)Math.Floor((double)num / size);
                }

                // Divide by normalized value
                input.Data<float>()[i] /= norm.Data<float>()[resultIndex];
            }

            return input;
        }

        private static int SafeIndex(int index, long length)
        {
            if (index < 0)
            {
                index += Convert.ToInt32(length);
            }
            if (index < 0 || index >= length)
            {
                throw new IndexOutOfRangeException("Index out of range");
            }
            return index;
        }
    }
}