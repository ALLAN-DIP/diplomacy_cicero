# AMR



Script to map single ENG sentence to DAIDE:

Download the pre-trained model file from:

https://drive.google.com/drive/folders/1buUmjbY3rRZDDNEBSTVh8wObZp38Gvy8?usp=sharing

and put the model under personal/SEN_REC_MODEL/ .

<details>
<summary>CLI examples for single message</summary>

```
python single.py --english "I propose ally between us" --sender "Russia" --recipient "Turkey"
``` 
</details>



<details>
<summary>CLI examples for multi messages</summary>

```
python multi-message.py --document msg_daide_state_AIGame_18.json
``` 
</details>

If you have a test file containing several AMRs, please refer to DAIDE/Diplomacy/README.md to parse it from AMR to DAIDE.

You can also train and test your own dataset of sentences and AMRs using the code below

make sure your dataset should under 

AMR/amrlib/amrlib/data/diplomacy/training/

AMR/amrlib/amrlib/data/diplomacy/dev/

AMR/amrlib/amrlib/data/diplomacy/test/

<details>
<summary>CLI examples_2</summary>

```
cd AMR/amrlib/scripts/33_Model_Parse_XFM
python 10_Collect_AMR_Data.py
python 20_Train_Model.py
python 22_Test_Model.py
``` 
</details>

Current Loaded Modules on CARC:
1) gcc/11.3.0   2) openblas/0.3.21   3) openmpi/4.1.4   4) pmix/3.2.3   5) usc/11.3.0 

For using AMR model to test on 300-ish gold DAIDE dataset, please refer to code here:https://github.com/ALLAN-DIP/diplomacy_cicero/blob/experiments/fairdiplomacy/AMR/test.py