# TODO

Explore possible tokenization strategies:
- [x] character level 
  - not sufficient
- [x] naive word level 
  - not sufficient
- [x] off-shelf tokenizers
  - allegro/herbert-klej-cased-tokenizer-v1 
    - 50560 vocab size 
    - good performance
  - maybe there is a better tokenizer on huggingface(?)
- [ ] train off-shelf tokenizer
- [ ] Tokenizer with ~32k vocab size 
  - [resource on why 32k](https://github.com/alasdairforsythe/tokenmonster/blob/main/benchmark/pretrain.md)
- [ ] Tokenizers with even lower vocab size 
  - might be better for this case