# BoNBoN

## Install Requirements

First, create a Python virtual environment.

```
conda create -n env python=3.10
conda activate env
```

Then, install all required packages.
```
pip install -r requirements.txt
git clone https://github.com/gl-ybnbxb/BoNBoN.git
cd BoNBoN
```
Also, follow the [official guideline](https://pytorch.org/get-started/locally/) to install PyTorch.

## Building Best-and-Worst Training Data

### Description
* **best_and_worst_of_n_sampler**
    * Inputs
    * Output: a dictionary of dictionaries
        * keys are $n$ in `n_seq`
        * each sub-dictionary is the best-and worst data for corresponding $n$
            * keys are prompt in `prompts`
            * `responses` are responses from the reference model
            * `pairs` records index pairs for each best-and-worst response pair. In each pair, the former one is the best of $n$ response and the latter one is the worst of $n$ sample.

### Usage

One example is [here](https://github.com/gl-ybnbxb/BoNBoN). Also see `build_data_main.py` for a python file example.

## Training Scirpts