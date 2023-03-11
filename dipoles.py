import torch
import numpy as np
import wandb
from tqdm import tqdm
import argparse
import os
import shutil
import random

from random import shuffle
from torch.optim import Adam, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from e3nn.nn.models.v2103.gate_points_networks import SimpleNetwork

from losses import implicit_score_matching
from data_manager import create_partitions, to_device, DeviceDataLoader, ImportData, MyLoader
from utils import init_weights, TranslateEnergies


def Train_routine(net, train_set, val_set, optimizer, wandb_flag, schedular, save_model_time, path_to_save, args):
    for epoch in range(args.numepochs):
        print(f"I am at epoch {epoch+1}")
        net.train()
        for batch_idx, batch in enumerate(tqdm(train_set)):
            optimizer.zero_grad()
            batch["pos"].requires_grad_()
            e = net(batch)
            scores = -torch.autograd.grad(outputs=e, inputs=batch["pos"], grad_outputs=torch.ones_like(e), retain_graph=True,create_graph=True)[0]
            scores = scores.reshape(len(batch),4,3)
            
            loss, loss1, loss2 = implicit_score_matching(batch["pos"], scores, args.batchsize)
            if wandb_flag==True:
                wandb.log({"train_loss": loss, "train_loss1":loss1, "train_loss2":loss2, "step":len(train_set)*epoch+len(train_set)*batch_idx/(len(train_set)+1) ,"epoch":epoch, "batch":batch_idx})
            loss.backward()
            if args.clip_norm==1:
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=args.clip_value, norm_type=2.0)
            else:
                torch.nn.utils.clip_grad_value_(parameters=net.parameters(), clip_value=args.clip_value)
            optimizer.step()
            optimizer.zero_grad()

        net.eval()
        val_epoch_loss=0
        for batch_idx, batch in enumerate(val_set):
            batch["pos"].requires_grad_()
            e = net(batch)
            scores = -torch.autograd.grad(outputs=e, inputs=batch["pos"], grad_outputs=torch.ones_like(e), retain_graph=True,create_graph=True)[0]
            scores = scores.reshape(len(batch),4,3)
            
            loss, loss1, loss2 = implicit_score_matching(batch["pos"], scores, args.batchsize)
            val_epoch_loss += loss
            if wandb_flag==True:
                wandb.log({"val_loss": loss, "val_loss1":loss1, "val_loss2":loss2, "step":len(val_set)*epoch+len(val_set)*batch_idx/(len(val_set)+1), "epoch":epoch, "batch":batch_idx})
        schedular.step(val_epoch_loss)
        
        if epoch == 0 or epoch%(save_model_time)==save_model_time-1:
            torch.save(net.state_dict(), os.path.join(path_to_save, f"tm_{epoch}.pt"))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", help="num layers", type=int, default=1, required=False)
    parser.add_argument("--lmax", help="num layers", type=int, default=2, required=False)
    parser.add_argument("--mul", help="multiplicity", type=int, required=True)
    parser.add_argument("--lr", help="learning rate", type=float, required=True)
    parser.add_argument("--cutoff", help="cutoff", type=float, required=True)
    parser.add_argument("--batchsize", help="batchsize", type=int, required=True)
    parser.add_argument("--numepochs", help="numepochs", type=int, required=True)
    parser.add_argument("--gamma", help="gamma scheduler", type=float, default=0.5, required=False)
    parser.add_argument("--seed", help="init seed", type=int, default=0, required=False)
    parser.add_argument("--clip_norm", help="1 if clip norm, 0 if clip value", type=int, required=True)
    parser.add_argument("--clip_value", help="grad clip value", type=float, required=True)
    parser.add_argument("--optim", help="optimizer", type=str, required=True)
    parser.add_argument("--wd", help="weight_decay", type=float, required=True)
    parser.add_argument("--edit", help="edit Data", action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_percent", help="use data's %", action=argparse.BooleanOptionalAction)
    parser.add_argument("--percent", help="what % use", type=int, required=True)
    parser.add_argument("--all_in_batch", help="True if put all datas single batch", action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb", help="True if save losses in w&b", action=argparse.BooleanOptionalAction)
    parser.add_argument("--save_model", help="Epochs after saving mdoel", type=int, required=True)
    parser.add_argument("--patience", help="Schedular patience", type=int, required=True)
    parser.add_argument("--cooldown", help="Schedular cooldown", type=int, required=True)
    
    args = parser.parse_args()

    
    
    print("Network setup")
    for a in args._get_kwargs():
        print(f"{a[0]}: {a[1]}")
        
    torch.manual_seed(args.seed) # 1 1 1 funziona
    random.seed(args.seed)
    np.random.seed(args.seed)

    path=f"./trained_model/lay={args.layers}_lmax={args.lmax}_mul={args.mul}_lr={args.lr}_batchsize={args.batchsize}_epochs={args.numepochs}_optim={args.optim}_seed={args.seed}_clip_norm={args.clip_norm}_clip_value={args.clip_value}_gamma={args.gamma}_percent={args.percent}_cutoff={args.cutoff}"
    
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device where I am running the code: {device}")        

    data = ImportData("dipoles_model/positions1.npy","dipoles_model/positions2.npy", args.edit, args.use_percent, args.percent)

    partition = create_partitions(data, train_size=0.80, val_size=0.2)

    train_set = MyLoader("train", partition, device, args.batchsize, args.all_in_batch)
    val_set = MyLoader("validation", partition, device, args.batchsize, args.all_in_batch)

    net = SimpleNetwork(
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
    
    net_params = {
        "irreps_in":"4x0e",
        "irreps_out":"1x0e",
        "max_radius":args.cutoff,
        "num_neighbors":3.0,
        "num_nodes": 4.0,
        "layers":args.layers,
        "mul":args.mul,
        "lmax":args.lmax,
        "pool_nodes":True
    }
    
    if args.wandb==True:
        wandb.login(key="b2d4378278d128tye38d9a6213g187fab529f895")
        wandb.init(project="my-Alvis-project-percent", config={**net_params, **{"learningrate":args.lr, "batchsize":args.batchsize,"numepochs":args.numepochs, "seed":args.seed, "optim":args.optim}})
        
    net.to(device)
    net.apply(init_weights)

    if args.optim=="Adam":
        optimizer = Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.optim=="RMSprop":
        optimizer = RMSprop(net.parameters(), lr=args.lr, weight_decay=args.wd)
    optimizer.zero_grad()
    schedular = ReduceLROnPlateau(optimizer, "min", factor=args.gamma, patience=args.patience, cooldown=args.cooldown, verbose=True, min_lr=1e-6)

    Train_routine(net, train_set, val_set, optimizer, args.wandb, schedular, args.save_model, path, args)
    print("Trained finished")

if __name__ == '__main__':
    main()
