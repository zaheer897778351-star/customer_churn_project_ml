import yaml
import os

def load_config():
    path = os.path.join("src","config","config.yaml")
    with open(path,"r") as file:
        config = yaml.safe_load(file)

    return config
