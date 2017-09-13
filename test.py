from Tiramisu import Tiramisu
import numpy as np
import matplotlib.pyplot as plt

tiramisu = Tiramisu()
model = tiramisu.model

model.load_weights('./weights/prop_tiramisu_weights_67_12_func_10-e7_decay150.hdf5', by_name=False)
