import torch

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    
    
def TranslateEnergies(energies):
    mean_value = 0
    for i in range(len(energies)):
        mean_value += energies[i]
    mean_value = mean_value/(len(energies))
    for i in range(len(energies)):
        energies[i] = energies[i] - mean_value
    return energies
