1. Use an open-source GPT-2/nanoGPT model to start:

GPT-2 small class (124M) or Pythia-160M equivalent

Training:
- Corpus: OpenWebText (or equivalent cleaned OWT)
- Context length: 1024
- Standard GPT-2 hyperparams (AdamW, cosine LR, etc.)

Validation:
- Match validation loss on OpenWebText within ~1-2%
- Reproduce benchmark with 1-2% perplexity: WikiText-103

2. Replace only the LayerNorm with something SMT encodable (DyT):

Show comparable performance

3. Replace only the attention mechanism with PWL or alpha-entmax attention:

Show comparable performance using same training process

4. Replace both LayerNorm and attention, test again

5. Encode the entire thing in an SMT prover

Attempt to prove simple statement about the entire model

6. Extract a circuit and prove properties about it (see README)

7. See if the properties generalize to the original Transformer model
