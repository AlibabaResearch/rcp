import os
import numpy as np
import torch

class Unbuffered:
    def __init__(self, stream, filename):
       self.stream = stream
       assert os.path.exists(os.path.dirname(filename))
       self.file = open(filename, "w")
       
    def write(self, data):
       self.stream.write(data)
       self.stream.flush()
       self.file.write(data)    # Write the data of stdout here to a text file as wel

    def flush(self):
        pass

    def __exit__(self, *args):
        self.file.close()
    
    def close(self):
        self.file.close()

def debugger_mode():
    import sys
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    elif gettrace():
        return True


def get_num_workers(num_workers):
    if debugger_mode():
        return 0
    else:
        return num_workers

def convert_dict_to_device(input, device):
    input_new = {}
    for key, value in input.items():
        input_new[key] = torch.tensor(value, dtype=torch.float32, device=device)
    return input_new

def detach_tensor(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x


def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

