namespace HuggingfaceOnnx.Models.MiniLM
{
    public class MiniLML6v2Config
    {
        public MiniLML6v2Config()
        {
            this.MaxSequenceLength = 512;
        }

        public int MaxSequenceLength { get; set; }
    }
}