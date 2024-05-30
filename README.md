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
    * Inputs:
        * `query_list`: a list of prompts
        * `batch_size`: batch size for data generation
        * `max_len`: the max number of new tokens for the reference model
        * `model`, `tokenizer`: the reference model for generation and its tokenizer
        * `rw_model`, `rw_tokenizer`: the reward model and its tokenizer
        * `device`: the device where the models are
        * `gen_kwargs`: some generation arguments
        * `n_seq`: a list of $n$, i.e., best of how many
    * Output: a dictionary of dictionaries
        * keys are $n$ in `n_seq`
        * each sub-dictionary is the best-and worst data for corresponding $n$
            * keys are prompt in `query_list`
            * `responses` are responses from the reference model
            * `pairs` records index pairs for each best-and-worst response pair. In each pair, the former one is the best of $n$ response and the latter one is the worst of $n$ sample.

### Usage

One python file example to build the data is [here](https://github.com/gl-ybnbxb/BoNBoN/blob/main/build_data/build_data_main.py).

## Training Scirpts


## Evaluations

See [kl_from_samples.py](https://github.com/gl-ybnbxb/BoNBoN/blob/main/metrics/kl_from_samples.py) and [eval_reward_model.py](https://github.com/gl-ybnbxb/BoNBoN/blob/main/metrics/eval_reward_model.py) for KL divergence and win rate computations.