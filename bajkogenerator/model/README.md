# TODO
### Models
implement text generation models:
- [x] lstm text generation
- [x] decoder only transformer
- [ ] encoder-decoder transformer
- [ ] encoder only transformer

### Sampling
Explore search and sampling methods:
- [x] multinomial sampling
- [x] greedy search
- [ ] ~~beam search~~
- [ ] ~~no-repeat ngrams~~
- [x] top-k
- [x] top-p
- [x] mix top-k + top-p

### Training
- [ ] Warmup
  - [ ] Learning rate scheduler
- [x] Gradient clipping
- [x] RAdam optimizer
    - very good results so far
    - might substitute warmup
- [x] Variable length sequences
    - not much improvement in LSTM