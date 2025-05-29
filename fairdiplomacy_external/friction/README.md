# Counterfactual RL against Deception (CTRL-D)

This is our team Cicero's repo corresponds to ACL findings paper "Should I Trust You? Detecting Deception in Negotiations using Counterfactual RL"

## 🚀 Getting Started

These instructions will help you set up and run the project on your local machine.

## 1. Running Cicero
Since CTRL-D depends highly on Cicero value model, this step is required. You have **two** options:

### 1) quick and easy by using docker/apptainer. This docker container is installed with Cicero's requiremented libraries with additions to run with our user interface from [Paquette](https://github.com/diplomacy/diplomacy) for internal use. 

```
docker pull ghcr.io/allan-dip/diplomacy_cicero:latest
```

### 2) install on your own machine, which you can follow steps on [Meta's Cicero github](https://github.com/facebookresearch/diplomacy_cicero)

## 2. Install AMR
To parse any messages in English to AMR, we recommend you install AMR following this [Diplomacy AMR github](https://github.com/YanzeJeremy/AMR.git), under the path `diplomacy_cicero/fairdiplomacy_external`

## 3. Now let's install BERT!
We run BERT using Python3.10 with these list of libralies
```
torch>=1.9.0         # For PyTorch
transformers>=4.0.0  # For HuggingFace BERT
scikit-learn>=0.24.0 # For StandardScaler, joblib, etc.
numpy>=1.19.0        # For array operations
joblib>=1.0.0        # For saving/loading models or scalers
```

## 4. It's time for detecting lies!
There are three main steps that the code in this folder could work to detect deception
### 1. `utils.py` 
With function `load_state_and_detect()` this parse an English message to AMR then extract into Diplomacy moves which we input to Cicero value model to calculate _bait_, _switch_ and _edge_

**Bait:** $$U_1 = u_i(\hat{a}_i, \hat{a}_j) - u_i(a_i, \hat{a}_j)$$

**Switch:** $$U_2 = u_i(\hat{a}_i, \hat{a}_j) - u_i(\hat{a}_i, a_j)$$

**Edge:** $$U_3 = u_j(\hat{a}_i, a_j) - u_j(a_i, a_j)$$

We have Peskoff et al, (2020) dataset available [here](https://drive.google.com/drive/folders/1q6osSBSTnzz5U6GNgZEFaoWLa52akka4?usp=drive_link) (need permission, email to wwongkam@umd.edu), and its 1000 samples with bait, switch and edge calculated [here](https://drive.google.com/file/d/1hF3vnRtHuPADPgVlBN3I-DZ2mGNMmkXI/view?usp=drive_link). 

### 2. `train_bert.py` and `eval_bert.py`
Training BERT is possible through `train_bert.py` where you can download corresponding files from [here](https://drive.google.com/drive/folders/1zvtZuuCjAlBeckLQh8etm34xE4CwXypF?usp=drive_link) to train.

In `eval_bert.py`, the file to evaluate also available [here](https://drive.google.com/drive/folders/1zvtZuuCjAlBeckLQh8etm34xE4CwXypF?usp=drive_link) with BERT model in a folder [score_bert_500](https://drive.google.com/drive/folders/1PvuNx06vnYKolp74bzCPBxrLy0qeVutU?usp=drive_link) and `scaler.pkl`

```bibtex
@misc{wongkamjan2025itrustyoudetecting,
  title={Should I Trust You? Detecting Deception in Negotiations using Counterfactual RL}, 
  author={Wichayaporn Wongkamjan and Yanze Wang and Feng Gu and Denis Peskoff and Jonathan K. Kummerfeld and Jonathan May and Jordan Lee Boyd-Graber},
  year={2025},
  eprint={2502.12436},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2502.12436}, 
}
