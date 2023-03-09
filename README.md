# FlowChain


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
python src/main --config_file config/TP/FlowChain/CIF_separate_cond_v_trajectron/eth.yml --mode train
```

You can also find our pretrained models [here](https://drive.google.com/drive/folders/1j1Uu0IX21D7UJ4b7M8FsClMk3l1YVzQe?usp=share_link)

# Testing
without visualization
```
python src/main --config_file config/TP/FlowChain/CIF_separate_cond_v_trajectron/eth.yml --mode test
```

with visualization
```
python src/main --config_file config/TP/FlowChain/CIF_separate_cond_v_trajectron/eth.yml --mode test
```
