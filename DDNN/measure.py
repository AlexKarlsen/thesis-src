import branchymodels
from op_counter import measure_model

model = branchymodels.BDenseNet.BDenseNet

flop, params = measure_model(model, 224, 224)