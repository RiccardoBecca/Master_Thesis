import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from e3nn.nn.models.v2103.gate_points_networks import SimpleNetwork
#from abc import ABC, abstractmethod
#from scipy.spatial.distance import cdist, pdist
from tqdm import tqdm
#import cProfile
import argparse
import os
import shutil
import wandb



def MyLoader(data_name, partition, device, batch_size):
    out=[]
    for item in partition[data_name]:
        pos=torch.from_numpy(item)
        pos = pos.to(torch.float32) #to change data type
        pos.requires_grad_()
        pos = pos.to(device)
        tempo = Data(pos=pos, x=torch.eye(len(pos), 4))
        out.append(tempo.to(device))
    return DataLoader(out, batch_size, drop_last=True, shuffle=True)


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
    
def get_my_sets(my_args, my_device, file1, file2):
    positions1 = np.load(file1)
    positions2 = np.load(file2)
    data = np.vstack((positions1, positions2))

    np.random.shuffle(data)
    partition = create_partitions(data, train_size=my_args.train_size, val_size=my_args.val_size)
    train_set = MyLoader("train", partition, my_device, my_args.batchsize)
    val_set = MyLoader("validation", partition, my_device, my_args.batchsize)
    
    return train_set, val_set
    
def get_my_paths(my_args):
    trained_path=f"./trained_model/lay={my_args.layers}_lmax={my_args.lmax}_mul={my_args.mul}_lr={my_args.lr}_batchsize={my_args.batchsize}_epochs={my_args.numepochs}_seed={my_args.seed}_gamma={my_args.gamma}_patience={my_args.patience}_cooldown={my_args.cooldown}_cutoff={my_args.cutoff}_wd={my_args.wd}_Tmax={my_args.T_max}_replicas={my_args.n_replicas}_noise={my_args.std_noise}_h={my_args.h}_epsilon={my_args.epsilon}"

    losses_path=f"./losses/lay={my_args.layers}_lmax={my_args.lmax}_mul={my_args.mul}_lr={my_args.lr}_batchsize={my_args.batchsize}_epochs={my_args.numepochs}_seed={my_args.seed}_gamma={my_args.gamma}_patience={my_args.patience}_cooldown={my_args.cooldown}_cutoff={my_args.cutoff}_wd={my_args.wd}_Tmax={my_args.T_max}_replicas={my_args.n_replicas}_noise={my_args.std_noise}_h={my_args.h}_epsilon={my_args.epsilon}"
        
        
    if os.path.exists(trained_path):
        shutil.rmtree(trained_path)
    os.mkdir(trained_path)
    if os.path.exists(losses_path):
        shutil.rmtree(losses_path)
    os.mkdir(losses_path)
    return trained_path, losses_path
    
def init_wandb(my_args):
    model_params = {
        "irreps_in":"4x0e",
        "irreps_out":"1x0e",
        "max_radius":my_args.cutoff,
        "num_neighbors":3.0,
        "num_nodes": 4.0,
        "layers":my_args.layers,
        "mul":my_args.mul,
        "lmax":my_args.lmax,
        "pool_nodes":True
    }

    wandb.login(key="b2d4365678d648fae38d9a6208e187fab529f895")
    wandb.init(project="ESVGD-Feb", config={**model_params, **{"learningrate":my_args.lr, "weight_decay":my_args.wd, "T_max":my_args.T_max, "Replicas":my_args.n_replicas, "Noise_std":my_args.std_noise, "gamma":my_args.gamma, "patience":my_args.patience, "cooldown":my_args.cooldown, "h_value":my_args.h, "epsilon":my_args.epsilon, "batchsize":my_args.batchsize,"numepochs":my_args.numepochs, "seed":my_args.seed}})
    
    
def get_replicas(batch, my_device, n_replicas, my_args):
    """This function is used to create four tensors which contains all the replicas of the system.
        entire_pos      contains the positions of the atom of each replica
        entire_x        contains the features of the atom of each replica (this is equal to the original batch["x"] of the molecule)
        entire_batch    assing to each replica a different batch value in order to calculate the energy
        entire_ptr      contains the beginning and the end of each block of replicas. The value =1 is assigned to all replicas of the first molecule, and so on
        
        Before the return the return, normal noise is addedd to entire_pos
    """
    last_batch=0
    last_ptr=0
    for i in range(len(batch["ptr"])-1):
        if i==0:
            entire_pos=(batch["pos"][batch["ptr"][i]:batch["ptr"][i+1]]).repeat(n_replicas,1).to(my_device)
            entire_x=(batch["x"][batch["ptr"][i]:batch["ptr"][i+1]]).repeat(n_replicas,1).to(my_device)
            entire_batch=((torch.arange(last_batch, last_batch+n_replicas, dtype=torch.int64).to(my_device)).repeat_interleave(batch["ptr"][i+1]-batch["ptr"][i])).to(my_device)
            last_batch+=n_replicas
            entire_ptr=torch.tensor([last_ptr, last_ptr+(batch["ptr"][i+1]-batch["ptr"][i])*n_replicas]).to(my_device)
            last_ptr+=(batch["ptr"][i+1]-batch["ptr"][i])*n_replicas
        else:
            entire_pos=torch.cat((entire_pos, (batch["pos"][batch["ptr"][i]:batch["ptr"][i+1]]).repeat(n_replicas,1).to(my_device)), dim=0) #add noise inside
            entire_x=torch.cat((entire_x, (batch["x"][batch["ptr"][i]:batch["ptr"][i+1]]).repeat(n_replicas,1).to(my_device)), dim=0)
            entire_batch=torch.cat((entire_batch, (torch.arange(last_batch, last_batch+n_replicas, dtype=torch.int64).to(my_device)).repeat_interleave(batch["ptr"][i+1]-batch["ptr"][i]).to(my_device)))
            last_batch+=n_replicas
            entire_ptr=torch.cat((entire_ptr, torch.tensor([last_ptr+(batch["ptr"][i+1]-batch["ptr"][i])*n_replicas]).to(my_device)))
            last_ptr+=(batch["ptr"][i+1]-batch["ptr"][i])*n_replicas
            
    entire_pos+=torch.normal(mean=torch.zeros(entire_pos.size()), std=(my_args.std_noise)*torch.ones(entire_pos.size())).to(my_device)
        
    return entire_pos, entire_x, entire_batch, entire_ptr
        
        
def simulate(entire_x, entire_pos, entire_batch, entire_ptr, my_model, n_replicas, my_args):
    """This function is used to simulate all the replicas. Pipeline:
        1. Loop over the time
        2. Each timestep calculate energies and forces of the whole replicas of every molecules. This is easily done with entire_x, entire_pos, entire_batch
        3. At each timestep, loop over the "block replicas". The first iteration move the replicas of the first molecule. The second iteration move the replicas of the second model and so on
        4. For each block of replicas kernels are calculated. Kernels are calculated as k(x_i, x_j)=k_{RBF}(\Psi(x_i), \Psi(x_j)) where \Psi(x)=||x-x_{COM}|| where x_{COM} is the center of mass of each replica. This is done cause we need equivariant kernels (this is invariant)
        5. After kernels are calculated, their gradients are calculated. It is "manually" done since I haven't find a way to do that using autograd. Since I am using k_{RBF} kernel, the gradient of the kernel is easy to get analitically
        6. Calculate the product between the kernel and the forces
        7. Sum over the raws to get the update step for the simulation
    """
    for _ in range(my_args.T_max):
        #calculate energy and forces
        energy_t=my_model({"x":entire_x, "pos":entire_pos, "batch":entire_batch})
        force_t = (-torch.autograd.grad(outputs=energy_t, inputs=entire_pos, grad_outputs=torch.ones_like(energy_t))[0])
        
        #Loop over "Block replicas"
        for i in range(len(entire_ptr)-1):
            #dim_mol tells the number of atoms in each molecule. It is just a useful for future calculation
            dim_mol=int((entire_ptr[i+1]-entire_ptr[i])/n_replicas)
            
            #curr_force select only the forces of the replicas we are watching at
            curr_force=(force_t[entire_ptr[i]:entire_ptr[i+1]]).view(1,n_replicas,-1)
            
            #curr_replicas select only the positions of the replicas we are watching at
            curr_replicas=entire_pos[entire_ptr[i]:entire_ptr[i+1]]
            
            # Calculation of the kernels
            # Get center of mass of the molecule and expand it can be summed to curr_replicas
            # com contains the center of masses, pos_com contains all the replicas translated by their center of masses
            # Psi contains the \Psi(x_i) value for every replica x_i
            com=torch.repeat_interleave(((curr_replicas.reshape(n_replicas, -1, 3)).sum(axis=-2))/dim_mol, dim_mol, dim=0)
            pos_com=curr_replicas-com
            Psi_matrix=torch.norm(pos_com.view(n_replicas,-1), dim=-1)
            # P is the matrix with dimensions (n_replicas x n_replicas) which contains all the \Psi differences between all the pair of replicas
            P=Psi_matrix[:,None]-Psi_matrix
            kernels=torch.exp(-(torch.abs(P))/(my_args.h))
            
            
            # Calculation of the gradients of the kernels.
            first_matrix=-2.*((pos_com/(torch.repeat_interleave(Psi_matrix , dim_mol, dim=0))[:,None]).view(n_replicas,-1))/(my_args.h)
            second_matrix=P*kernels
            kernel_grad=(first_matrix[None, :,:]*(second_matrix.T)[:,:,None]).view(n_replicas, n_replicas, dim_mol,3)
                        
            #Calculate the product between kernels and forces
            KF=(kernels[:,:,None]*curr_force).view(n_replicas, n_replicas, -1,3)
            
            # Sum the rows to get the update
            update=((kernel_grad+KF).sum(axis=1)).view(-1,3)
            
            #update the positions
            entire_pos[entire_ptr[i]:entire_ptr[i+1]]+=(my_args.epsilon)*(update)/n_replicas
            
    return entire_pos
    
    
def Train_Routine(my_args, my_train_set, my_val_set, my_device, my_model, my_optimizer, my_schedular, my_trained_path, my_losses_path):

    n_replicas=my_args.n_replicas
    my_model.train()
    for epoch in range(my_args.numepochs):
    
    
    
        #################### TRAINING ####################
        train_epoch_loss=0
        for batch_idx,batch in (enumerate(tqdm(my_train_set))):
            
            # See get_replicas comment
            entire_pos, entire_x, entire_batch, entire_ptr = get_replicas(batch, my_device, n_replicas, my_args)
            
            # Simulate replicas. See simulate comments
            entire_pos=simulate(entire_x, entire_pos, entire_batch, entire_ptr, my_model, n_replicas, my_args)
            
            fake_energy=my_model({"x":entire_x, "pos":entire_pos, "batch":entire_batch})
            true_energy=my_model(batch)
            
            loss=true_energy.mean()-fake_energy.mean()
            
            if my_args.wandb==True:
                wandb.log({"train_loss": loss, "step":len(my_train_set)*epoch+len(my_train_set)*batch_idx/(len(my_train_set)+1) ,"epoch":epoch, "batch":batch_idx})
            
            torch.autograd.backward(loss)
                    
            my_optimizer.step()
            my_optimizer.zero_grad()
            train_epoch_loss+=loss
            
            with open(os.path.join(my_losses_path, "train_loss.txt"), 'a+') as f:
                f.write(str(float(loss))+'\n')
          
        with open(os.path.join(my_losses_path, "train_epoch_loss.txt"), 'a+') as f:
            f.write(str(float(train_epoch_loss))+'\n')
            
            
        
        #################### VALIDATION ####################
        val_epoch_loss=0
        for batch_idx,batch in (enumerate(my_val_set)):
        
            # Get the replicas
            entire_pos, entire_x, entire_batch, entire_ptr = get_replicas(batch, my_device, n_replicas, my_args)
            
            # Simulate replicas
            entire_pos=simulate(entire_x, entire_pos, entire_batch, entire_ptr, my_model, n_replicas, my_args)
                        
            fake_energy=my_model({"x":entire_x, "pos":entire_pos, "batch":entire_batch})
            true_energy=my_model(batch)
            
            loss=true_energy.mean()-fake_energy.mean()
            
            
            if my_args.wandb==True:
                wandb.log({"val_loss": loss, "step":len(my_val_set)*epoch+len(my_val_set)*batch_idx/(len(my_val_set)+1) ,"epoch":epoch, "batch":batch_idx})
            
            
            with open(os.path.join(my_losses_path, "val_loss.txt"), 'a+') as f:
                f.write(str(float(loss))+'\n')
            val_epoch_loss+=loss
        with open(os.path.join(my_losses_path, "val_epoch_loss.txt"), 'a+') as f:
            f.write(str(float(val_epoch_loss))+'\n')
           
        my_schedular.step(torch.abs(val_epoch_loss))
        
        if epoch == 0 or epoch%(my_args.save_model_time)==my_args.save_model_time-1:
            torch.save(my_model.state_dict(), os.path.join(my_trained_path, f"tm_{epoch}.pt"))



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", help="batchsize", type=int, required=True)
    parser.add_argument("--numepochs", help="numepochs", type=int, required=True)
    parser.add_argument("--lr", help="learning rate", type=float, required=True)
    parser.add_argument("--wd", help="weight_decay", type=float, required=True)
    parser.add_argument("--layers", help="num layers", type=int, required=True)
    parser.add_argument("--lmax", help="num layers", type=int, required=True)
    parser.add_argument("--mul", help="multiplicity", type=int, default=8 ,required=False)
    parser.add_argument("--cutoff", help="cutoff", type=float, required=True)
    parser.add_argument("--T_max", help="maximum T for the simulation", type=int, required=True)
    parser.add_argument("--n_replicas", help="number of replicas to be simulated", type=int, required=True)
    parser.add_argument("--std_noise", help="std for the gaussian noise", type=float, required=True)
    parser.add_argument("--gamma", help="gamma scheduler", type=float, required=True)
    parser.add_argument("--patience", help="Schedular patience", type=int, required=True)
    parser.add_argument("--cooldown", help="Schedular cooldown", type=int, required=True)
    parser.add_argument("--h", help="Bandwidth", type=float, required=True)
    parser.add_argument("--epsilon", help="step size for the simulation", type=float, required=True)
    parser.add_argument("--seed", help="init seed", type=int, required=True)
    parser.add_argument("--save_model_time", help="save_model_every", type=int, required=True)
    parser.add_argument("--wandb", help="True if save losses in w&b", action=argparse.BooleanOptionalAction)
    parser.add_argument("--train_size", help="trainset percent", type=float, default=0.8, required=False)
    parser.add_argument("--val_size", help="valset percent", type=float, default=0.2, required=False)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device where I am running the code: {device}")



    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # paths used to save losses and models
    trained_path, losses_path = get_my_paths(args)
    
    # get train and val set
    train_set, val_set = get_my_sets(args,device, "./dipoles_model/positions1.npy", "./dipoles_model/positions2.npy")


    #definition of the network
    model = SimpleNetwork(
        irreps_in="4x0e",
        irreps_out="1x0e",
        max_radius=args.cutoff,
        num_neighbors=3.0,
        num_nodes=4.0,
        layers=args.layers,
        lmax=args.lmax,
        mul=args.mul,
        pool_nodes=True
    )


    model.to(device)

    if args.wandb==True:
        init_wandb(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    optimizer.zero_grad()
    schedular = ReduceLROnPlateau(optimizer, "min", factor=args.gamma, patience=args.patience, cooldown=args.cooldown, verbose=True, min_lr=1e-6)

    # beginning of the training routine
    Train_Routine(args, train_set, val_set, device, model, optimizer, schedular, trained_path, losses_path)

if __name__ == '__main__':
    main()
