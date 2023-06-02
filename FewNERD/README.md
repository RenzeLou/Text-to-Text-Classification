
```bash
sh scripts/run_gen.sh 1 64 t5-small 5e-4 supervised
sh scripts/run_gen.sh 1 32 t5-base 5e-4 supervised
sh scripts/run_gen.sh 1 16 t5-large 5e-4 supervised
sh scripts/run_gen.sh 2 4 t5-3b 5e-4 supervised
```

```bash
sh scripts/run_gen.sh 0 64 t5-small 5e-4 inter
sh scripts/run_gen.sh 0 32 t5-base 5e-4 inter
sh scripts/run_gen.sh 0 16 t5-large 5e-4 inter
sh scripts/run_gen.sh 3 4 t5-3b 5e-4 inter
```

```bash
sh scripts/run_gen.sh 7 64 t5-small 5e-4 intra
sh scripts/run_gen.sh 7 32 t5-base 5e-4 intra
sh scripts/run_gen.sh 7 16 t5-large 5e-4 intra
sh scripts/run_gen.sh 7 4 t5-3b 5e-4 intra
```

TODO:
 - now uploaded the code for token-level classification (see `run_cls_token_level.py`), but it still can not work. The main issue is that, I am not sure what kind of data file this script supports. It asked to use 'csv' or 'json', but I don't know what's the input and output.