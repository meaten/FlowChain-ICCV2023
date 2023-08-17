# FlowChain
The repository contains the code for [Fast Inference and Update of Probabilistic Density Estimation on Trajectory Prediction](url) by Takahiro Maeda and Norimichi Ukita.

# Dependencies
```
pip install -r requirements.txt
```

# Data Preparation
```
python src/data/TP/process_data.py
```

# Model Training
For example of ETH split,
```
python src/main.py --config_file config/TP/FlowChain/CIF_separate_cond_v_trajectron/eth.yml --mode train
```

You can also find our pretrained models [here](https://drive.google.com/drive/folders/1bA0ut-qrgtr8rV5odUEKk25w9I__HjCY?usp=share_link)

Just download the 'output' folder to the root of this repo, and you are ready to test these models.

# Testing
without visualization
```
python src/main.py --config_file config/TP/FlowChain/CIF_separate_cond_v_trajectron/eth.yml --mode test
```

with visualization
```
python src/main.py --config_file config/TP/FlowChain/CIF_separate_cond_v_trajectron/eth.yml --mode test --visualize
```
