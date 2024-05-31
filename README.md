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

## Training Scripts

We are following the same training steps described in the original [DPO code repository](https://github.com/eric-mitchell/direct-preference-optimization/tree/main).

Step 1: Running SFT

Run SFT for Pythia 2.8B on Anthropic-HH data with batch size 64:
```
python -u train.py model=pythia28 datasets=[hh] loss=sft exp_name=sft_pythia28_AntrophicHH gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16
```

Step 2: Running DPO / IPO / BoNBoN

There are three types of losses implemented: `dpo`, `ipo`, and `bonbon` loss.

* Running DPO

```
python -u train.py model=pythia28 datasets=[hh] loss=dpo loss.beta=0.1 exp_name=anthropic_dpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/path/to/sft/LATEST/policy.pt
```

For running IPO please change the loss in the command above to `loss=ipo`.

* Running BoNBoN on best-of-n data:

Inside `preference_datasets.py` the `get_best_worst` function loads the best-and-worst-of-n data. Please make sure to change line 195 in this file to point to the location of your best-and-worst-of-n data.

For training BoNBoN please make sure to specify `loss=bonbon`, `loss.beta=beta_value` and `loss.alpha=alpha_value` as in the command below:  

```
python -u train.py model=pythia28 datasets=[hh_subset] loss=bonbon loss.beta=0.0275482094 loss.alpha=0.005 exp_name=anthropic_dpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/path/to/sft/LATEST/policy.pt
```

We run each model for 20k steps, then sample and evaluate from the trained model.

## Sampling from trained checkpoint


## Evaluations

See [kl_from_samples.py](https://github.com/gl-ybnbxb/BoNBoN/blob/main/metrics/kl_from_samples.py) and [eval_reward_model.py](https://github.com/gl-ybnbxb/BoNBoN/blob/main/metrics/eval_reward_model.py) for KL divergence and win rate computations.
