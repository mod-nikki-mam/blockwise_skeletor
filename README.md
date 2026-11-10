# testing blockwise ternary distillation technique
hobby project

## model deets
  - the model "skeletor" is first trained normally(full float quantization) in train.py, then distilled into ternary in distill.py
  - the model is a simple MLP classifier trained on the AG news dataset, uses Haar transforms instead of classical attention
