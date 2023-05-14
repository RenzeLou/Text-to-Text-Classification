## Environment

- Python 3.8.0
- PyTorch 2.0.0
- CUDA 11.7
- Transformers 4.27.4

Prepare the anaconda environment:

```bash
conda create -n t2t python=3.8.0
conda activate t2t
pip install -r requirements.txt
```

## Run the experiments

### 1. Prepare the data

In this experiment, we use [*banking_data*](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data) for intent classification. The data is already included in the repository.

Run the following command to preprocess the data:

```bash
python data_process.py
```

It will process the origin `train.csv` and `test.csv` into two parts: one is for text-to-text generation, where the labels are the original categories; while the other is for text-to-indices classification, where the labels are all converted to indices.   

Add `--split` to randomly split 10% samples from training set as validation set to tune the hyperparameters (we have already done it and fixed all the hyperparameters).

### 2. Train the model

For text-to-text generation, run the following command:

```bash
sh scripts/run_sen_gen.sh [GPU] [batch_size]
```

For text-to-indices classification, run the following command:

```bash
sh scripts/run_sen_cls.sh [GPU] [batch_size]
```

The results will be saved in `./out` folder.

### 3. Results

The results are shown in the following table:

<table style="height: 90px;" width="599">
<tbody>
<tr style="height: 18px;">
<td style="height: 36px; width: 128.219px;" rowspan="2"><strong>Model</strong></td>
<td style="text-align: center; height: 18px; width: 144.25px;"><strong>Text-to-Indices</strong></td>
<td style="text-align: center; height: 18px; width: 304.531px;" colspan="2"><strong>Text-to-Text</strong></td>
</tr>
<tr style="height: 18px;">
<td style="text-align: center; height: 18px; width: 144.25px;"><strong>accuracy</strong></td>
<td style="text-align: center; height: 18px; width: 128.219px;"><strong>accuracy</strong></td>
<td style="text-align: center; height: 18px; width: 170.312px;"><strong>In-distribution ratio</strong></td>
</tr>
<tr style="height: 18px;">
<td style="height: 18px; width: 128.219px;">T5-base (220M)</td>
<td style="text-align: center; height: 18px; width: 144.25px;">93.9935</td>
<td style="text-align: center; height: 18px; width: 128.219px;">93.7013</td>
<td style="text-align: center; height: 18px; width: 170.312px;">99.9675</td>
</tr>
<tr style="height: 18px;">
<td style="height: 18px; width: 128.219px;">T5-large (770M)</td>
<td style="text-align: center; height: 18px; width: 144.25px;">93.2143</td>
<td style="text-align: center; height: 18px; width: 128.219px;">93.7662</td>
<td style="text-align: center; height: 18px; width: 170.312px;">99.9351</td>
</tr>
<tr style="height: 18px;">
<td style="height: 18px; width: 128.219px;">T5-3B</td>
<td style="text-align: center; height: 18px; width: 144.25px;">94.4156</td>
<td style="text-align: center; height: 18px; width: 128.219px;">93.7987</td>
<td style="text-align: center; height: 18px; width: 170.312px;">99.9351</td>
</tr>
</tbody>
</table>


<!-- **TODO**:
- check the reason of the wired result of text-to-indices T5-3B. -->