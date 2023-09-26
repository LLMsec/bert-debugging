# bert-debugging


https://docs.google.com/document/d/1QzES2769WJxeJ1zoBxf1Atua79Ao5u1bMBgzy4sqf2k/edit

```
Bert(
  (embeddings): ModuleDict(
    (token): Embedding(30522, 128, padding_idx=0)
    (position): Embedding(512, 128)
    (token_type): Embedding(2, 128)
  )
  (ln): LayerNorm()
  (dropout): Dropout(p=0.9, inplace=False)
  (layers): ModuleList(
    (0-1): 2 x Layer(
      (query): Linear(in_features=128, out_features=128, bias=True)
      (key): Linear(in_features=128, out_features=128, bias=True)
      (value): Linear(in_features=128, out_features=128, bias=True)
      (dropout): Dropout(p=0.9, inplace=False)
      (attn_out): Linear(in_features=128, out_features=128, bias=True)
      (ln1): LayerNorm()
      (mlp): MLP(
        (dense_expansion): Linear(in_features=128, out_features=512, bias=True)
        (dense_contraction): Linear(in_features=512, out_features=128, bias=True)
      )
      (ln2): LayerNorm()
    )
  )
  (pooler): Sequential(
    (dense): Linear(in_features=128, out_features=128, bias=True)
    (activation): Tanh()
  )
)
```
