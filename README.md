# FAST
- Code for paper named: Decoding Covert Speech from EEG Using a Functional Areas Spatio-Temporal Transformer (FAST), which is currently under review
- This codebase is for reproducing the result on the publicly available dataset BCI Competition 2020 Track #3: Imagined Speech Classification (BCIC2020Track3)
- Contact: James Jiang Muyun (james.jiang@ntu.edu.sg)

## Dataset Preparation
1. Download the dataset from [https://osf.io/pq7vb/](https://osf.io/pq7vb/).
2. Place the dataset in the `BCIC2020Track3/` directory, the correct file structure should be the same as below:
```
BCIC2020Track3/
├── Training set/
│   ├── Data_Sample01.mat
│   ├── Data_Sample02.mat
....
├── Validation set/
│   ├── Data_Sample01.mat
│   ├── Data_Sample02.mat
...
```

## Data Preprocessing
Run the following command to preprocess the data:
```bash
python BCIC2020Track3_preprocess.py
```
The processed data will be saved in `Processed/BCIC2020Track3.h5`

## Training the Model
- To train the transformer model, run:
```bash
python train.py --model transformer --accelerator gpu --gpu 0 --folds "0-15"
```

- For Mamba2 (CUDA required), run:
```bash
python train.py --model mamba2 --accelerator gpu --gpu 0 --folds "0-15"
```

- To sweep all configs:
```bash
bash sweep.sh
```

After training, results are saved under `Results/FAST-<model>-spatial_projection-<True|False>/` with per-fold CSVs and `accuracies.csv`.

# Cite
Please cite our paper if you find this work is useful to you:
```
@article{jiang2025decoding,
  title={Decoding Covert Speech from EEG Using a Functional Areas Spatio-Temporal Transformer},
  author={Jiang, Muyun and Ding, Yi and Zhang, Wei and Teo, Kok Ann Colin and Fong, LaiGuan and Zhang, Shuailei and Guo, Zhiwei and Liu, Chenyu and Bhuvanakantham, Raghavan and Sim, Wei Khang Jeremy and others},
  journal={arXiv preprint arXiv:2504.03762},
  year={2025}
}
```
