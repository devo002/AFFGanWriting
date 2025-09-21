import torch

# Load the state dict
state_dict = torch.load('/home/vault/iwi5/iwi5333h/oldmodels_logs/29-07-25/contran-2900.model')

# Print all layer names
for key in state_dict.keys():
    print(key)