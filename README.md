>📋  A template README.md for code accompanying a Machine Learning paper
# Topological Anomaly Quantification for Semi-supervised Graph Anomaly Detection

## Overview
Semi-supervised graph anomaly detection identifies nodes deviating from normal patterns using a limited set of labeled nodes. This paper specifically addresses the challenging scenario where only normal node labels are available. To address the challenge of anomaly scarcity in real-world graphs, generative-based methods synthesize anomalies by linear/non-linear interpolation or random noise perturbation. However, these methods lack a quantitative assessment of anomalies, hindering the reliability of the generated ones. To overcome this limitation, we propose a generative graph anomaly detection model based on topological anomaly quantification (TAQ-GAD). First, TAQ-GAD designs a topological anomaly quantification module (TAQ), which quantifies node abnormality through two topological metrics: The node boundary score (NBS) quantifies the boundaryness of a node by evaluating its connectivity to labeled normal neighbors. The node isolation score (NIS) assesses the structural isolation of a node by evaluating its connection strength to other nodes within the same category. This anomaly measurement module dynamically screens unlabeled nodes with high anomaly scores as pseudo-anomaly nodes. Subsequently, the topological anomaly enhancement (TAE) module generates virtual anomaly center nodes and constructs their topological relationships with other nodes. Finally, the method integrates normal and pseudo-anomaly nodes on the enhanced graph for model training. Extensive experiments on benchmark datasets demonstrate TAQ-GAD’s superiority over state-of-the-art methods and effectively improve anomaly detection performance.
 
## Requirements

To install requirements:
```setup
pip install -r requirements.txt
```

[//]: # (Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...)
>📋
 TAQ-GAD is implemented in PyTorch 1.11.0 with Python 3.8, and all the experiments are run on a 24-core CPU with CUDA 11.3.

## Training
We provide the test code. We will release the training code after the paper is accepted.
python TAQ_run.py



## Datasets
For convenience, some datasets can be obtained from [google drive link](https://drive.google.com/drive/folders/1rEKW5JLdB1VGwyJefAD8ppXYDAXc5FFj?usp=sharing.). 
We sincerely thank the researchers for providing these datasets.
Due to the Copyright of DGraph-Fin, you need to download from [DGraph-Fin](https://dgraph.xinye.com/introduction).

| Dataset | Type                | Nodes     | Edges      | Attributes | Anomalies(Rate) |
|--------|---------------------|-----------|------------|------------|-----------------|
|Amazon | Co-review           | 11,944    | 4,398,392  | 25         | 821(6.9%)       |
|T-Finance| Transaction         | 39,357 2  | 1,222,543  | 10         | 1,803(4.6%)     |
|Reddit| Social Media        | 10,984    | 168,016    | 64         | 366(3.3%)       |
|Elliptic| Bitcoin Transaction | 46,564    | 73,248     | 93         | 4,545(9.8%)     |
|Photo| Co-purchase         | 7,535     | 119,043    | 745        | 698(9.2%)       |
|DGraph| Financial Networks  | 3,700,550 | 73,105,508 | 17         | 15,509(1.3%)    |
