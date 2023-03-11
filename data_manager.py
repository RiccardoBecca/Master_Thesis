import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def create_partitions(data, train_size, val_size):
    partition = {}
    train_n = int(len(data) * train_size)
    val_n = int(len(data) * val_size)
    test_n = len(data) - train_n - val_n

    train, val, test = torch.utils.data.random_split(data, [train_n, val_n, test_n], generator=torch.Generator().manual_seed(25))
        
    partition["train"] = train
    partition["validation"] = val
    partition["test"] = test
    return partition
    
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device, non_blocking=True)
    
    
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)
            
    def __len__(self):
        """Number of batches"""
        return len(self.dl)
       
       
def ImportData(name1, name2, edit, use_percent, percent):
    positions1 = np.load(name1)
    positions2 = np.load(name2)
    data = np.vstack((positions1, positions2))
    if edit==True:
        data = data[np.linalg.norm(data[:,-1,:] - data[:,-2,:], axis=-1) <= 5.0] #This line is to
    np.random.shuffle(data)
    if use_percent==True:
        print(f"Watch out! You are using {percent}% of the available dataset")
        new_data=[]
        new_len=int(len(data)*percent/100 +0.5)
        for i in range(new_len):
            new_data.append(data[i])
        return np.asarray(new_data)
    else:
        return data
    

def MyLoader(data_name, partition, device, batch_size, all_in_batch):
    out=[]
    for item in partition[data_name]:
        pos=torch.from_numpy(item)
        pos = pos.to(torch.float32) #to change data type
        x = torch.eye(4)
        
        pos = pos.to(device)
        x = x.to(device)
        tempo = Data(pos=pos, x=torch.eye(4))
        out.append(tempo.to(device))
    if all_in_batch==True:
        return DataLoader(out, len(out), drop_last=True)
    else:
        return DataLoader(out, batch_size, drop_last=True, shuffle=True)
