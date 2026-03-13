# a2d_qwen2_tinystories_200M

A2D-Qwen2 200M config for TinyStories n-gram analysis.
Same architecture as `a2d_qwen2_200M` (24 layers, hidden_size=768),
with the tokenizer swapped to match the n-gram statistics paper
(arxiv 2407.12034).

## Tokenizer setup (required before training)

Download the SentencePiece model from GCS and place it here:

```bash
pip install gcsfs
python -c "
import gcsfs
fs = gcsfs.GCSFileSystem('transformer-ngrams')
with fs.open('gs://transformer-ngrams/32768.model', 'rb') as f:
    data = f.read()
with open('tokenizer.model', 'wb') as f:
    f.write(data)
"
```

The file must be named `tokenizer.model` in this directory.

## Key differences from a2d_qwen2_200M

| Field          | a2d_qwen2_200M | a2d_qwen2_tinystories_200M |
|----------------|----------------|----------------------------|
| vocab_size     | 2202           | 32769 (32768 + mask token) |
| mask_token_id  | 2201           | 32768                      |
| bos_token_id   | 2              | 1                          |
| eos_token_id   | 3              | 2                          |
| pad_token_id   | 1              | 0 (unk)                    |
| tokenizer      | custom BPE     | SentencePiece 32768        |
