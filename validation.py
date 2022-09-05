import torch
import torch.nn as neural_net
from sklearn.model_selection import KFold


criterion = neural_net.CrossEntropyLoss()

torch.manual_seed(42)

num_epochs = 80

k = 10

splits = KFold(n_splits = k, shuffle = True, random_state = 42)

foldperf = {}
