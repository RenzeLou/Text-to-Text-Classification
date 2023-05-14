# Tk-Instruct

The code is based on the [Tk-Instruct](https://github.com/yizhongw/Tk-Instruct).

## Requirements

Our main experiments and analysis are conducted on the following environment:

- CUDA (11.3)
- cuDNN (8.2.0.53)
- Pytorch (1.10.0)
- Transformers (4.17.0)
- DeepSpeed

You can refer to the [Dockerfile](Dockerfile) for setting up the environment and install the required python libraries by running

```bash
pip install -r requirements.txt
```

Note: after the main exploration with 3B model, we train our 11B model on TPUs using the T5 code [here](https://github.com/google-research/text-to-text-transfer-transformer).

## Data

Our models are trained and evaluated on [Super-NaturalInstructions](https://github.com/allenai/natural-instructions), which can be cloned by running:

```bash
git clone git@github.com:allenai/natural-instructions.git data
```

## Training

Default T5 encoder-decoder (generator):

```bash
sh scripts/train_generator.sh 6 4 t5-3b 0  ## GPU, batch size, model name, whether mixing generation tasks (0/1)
```

T5 encoder-classifier (binary single-label classifier):

```bash