#updated
import os
import random
import torch.optim as optim
import numpy as np
import torch
import yaml
import csv
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb
import torch.nn.functional as F
from pathlib import Path
from custom_dataset_bg import customdataset
pretrain_demucs = True
if pretrain_demucs == True:
    from model_for_demucs import CosNetwork
    from model_for_demucs import load_pretrain,center_trim,normalize_input,unnormalize_input
    print("pretrained_model")
else:
    from model import CosNetwork
    from model import load_pretrain,center_trim,normalize_input,unnormalize_input
    print("no pretrained model")
start_epoch = 0

# Ensure deterministic behavior
#torch.backends.cudnn.deterministic = True
#random.seed(hash("setting random seeds") % 2**32 - 1)
#np.random.seed(hash("improves reproducibility") % 2**32 - 1)
#torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
#torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)
use_cuda = torch.cuda.is_available()
# Device configuration
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)
# remove slow mirror from list of MNIST mirrors
yaml_file_path = Path(__file__).resolve().parent.parent / 'constants.yaml'
with open(yaml_file_path, "r") as file:
        config_yaml = yaml.safe_load(file)
yaml_train_dir = config_yaml["train_folder_Dataloader"]
yaml_pretrain_path = config_yaml["pretrain_path"]
yaml_checkpoints_dir = config_yaml["checkpoints_path"]
yaml_learning_rate = config_yaml["learning_rate"]
yaml_start_epoch = config_yaml["start_epoch"]
yaml_batch_size = config_yaml["batch_size"]
yaml_epochs = config_yaml["epochs"]
name = "multimic_experiment"
wandb.login()
#lr
#BatchSize
#epochs
#n_mics
#sr
#perturb_prob
#mic_radius
print("test saving")
config = dict(
    epochs=yaml_epochs,
    batch_size=8,
    learning_rate=yaml_learning_rate,
    n_mics = 6,
    sr = 44100,
    name="true-frog-8",
    pretrain_path = None,
    perturb_prob = 1.0,
    mic_radius = 0.0463,
    num_workers = 20,
    dataset="VCTK",
    architecture="Cos_1610_bg_More")
print(config)
def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(project="Cos_1610_bg_More", config=hyperparameters):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, train_loader, test_loader, val_loader, criterion, optimizer = make(config)
      print(model)

      # and use them to train the model
      train(model, train_loader, val_loader, criterion, optimizer, config)

      # and test its final performance
      test(model, test_loader)

    return model
def make(config):
    # Make the data
    train = get_data(train=True)
    val = get_data(val=True)
    test = get_data(test=True)
    train_loader = make_loader(train, batch_size=config.batch_size,num_workers=config.num_workers)
    test_loader = make_loader(test, batch_size=config.batch_size,num_workers=config.num_workers)
    val_loader = make_loader(val, batch_size=config.batch_size, num_workers=config.num_workers)
    # Make the model
    model = CosNetwork().to(device)

    if pretrain_demucs:
        url = 'https://dl.fbaipublicfiles.com/demucs/v3.0/demucs-e07c671f.th'#pretrain demucs
        state_dict = torch.hub.load_state_dict_from_url(url)
        load_pretrain(model, state_dict)
        print("pretrained loaded")
    #if start_epoch>0:
    if False:
        checkpoints_dir = "checkpoints/cos"
        name = "multimic_experiment"
        print("hello")
        checkpoints_dir = os.path.join(checkpoints_dir,name)
        #checkpoint_path = os.path.join(checkpoints_dir, "{}_Cos_bg_0610.pt".format(int(start_epoch) - 1))
        checkpoint_path = os.path.join(checkpoints_dir, "{}_Cos_bg_0610.pt".format(42))
        print(checkpoint_path)
        state_dict = torch.load(checkpoint_path)
        load_pretrain(model, state_dict)    

    # Make the loss and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    
    return model, train_loader, test_loader ,val_loader,  criterion, optimizer
def get_data(train=False, val=False, test=False):
    # Define paths for train, validation, and test directories

    train_dir = Path("../gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN/More_bg_3d_1610")
    val_dir = Path("../gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN/Val_bg_3d_0610")
    test_dir = Path("../gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN/Test_bg_3d_0610")
    # Select the appropriate directory based on the flag
    if train:
        data_dir = train_dir
    elif val:
        data_dir = val_dir
    elif test:
        data_dir = test_dir
    # Create the customdataset using the selected directory
    split_dataset = customdataset(data_dir, n_mics=6, sr=44100, perturb_prob=1.0, mic_radius=0.0463)
    
    return split_dataset

def make_loader(dataset, batch_size,num_workers):
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': True
        } if use_cuda else {}
    loader = DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         **kwargs)
    return loader
def train(model, loader,val_loader, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    val_loss_file = os.path.join(yaml_checkpoints_dir, name, "validation_losses_2.csv")
    
    # Write the header to the CSV file
    with open(val_loss_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "validation_loss"])
    
    
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    print("hello")
    val_loss = 0
    for epoch in tqdm(range(config.epochs)):
        model.eval()
        with torch.no_grad():
            for data, label_voice_signals, window_idx in val_loader:
                data = data.to(device)
                label_voice_signals = label_voice_signals.to(device)
                window_idx = window_idx.to(device)

                data, means, stds = normalize_input(data)
                valid_length = model.valid_length(data.shape[-1])
                delta = valid_length - data.shape[-1]
                padded = F.pad(data, (delta // 2, delta - delta // 2))
                output_signal = model(padded, window_idx)
                output_signal = center_trim(output_signal, data)
                output_signal = unnormalize_input(output_signal, means, stds)
                output_voices = output_signal[:, 0]
                loss = criterion(output_voices, label_voice_signals)
                val_loss += loss.item()


        val_loss = val_loss / len(val_loader)

        model.train()
        for _, (data, label_voice_signals,window_idx) in enumerate(loader):

            loss = train_batch(data, label_voice_signals,window_idx, model, optimizer, criterion)
            example_ct +=  len(data)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 5) == 0:
                train_log(loss, example_ct,batch_ct, epoch,val_loss)

        with open(val_loss_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, val_loss])
        
        print(f"Validation Loss after epoch {epoch}: {val_loss:.5f}")
        if epoch % 1 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(yaml_checkpoints_dir,name,"{}_Cos_bg_More_1610.pt".format(epoch+start_epoch))
                )
    
def train_batch(data, label_voice_signals,window_idx, model, optimizer, criterion):
    data = data.to(device)
    label_voice_signals = label_voice_signals.to(device)
    window_idx = window_idx.to(device)
    # Forward pass ➡
    data, means, stds = normalize_input(data)
    # Reset grad
    optimizer.zero_grad()

    # Run through the model
    valid_length = model.valid_length(data.shape[-1])
    delta = valid_length - data.shape[-1]
    padded = F.pad(data, (delta // 2, delta - delta // 2))
    output_signal = model(padded, window_idx)
    output_signal = center_trim(output_signal, data)
    # Un-normalize
    output_signal = unnormalize_input(output_signal, means, stds)
    output_voices = output_signal[:, 0]

    loss = criterion(output_voices, label_voice_signals)
    
    # Backward pass ⬅
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    # Step with optimizer
    optimizer.step()

    return loss
def train_log(loss, example_ct,batch_ct, epoch,val_loss):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss,"val_loss": val_loss}, step=batch_ct)
    #wandb.log({"loss_in_batches": loss}, step=batch_ct)
    print(f"Loss after {str(example_ct).zfill(7)} examples: {loss:.4f}")
def test(model, test_loader):
    model.eval()

    # Run the model on some test examples
    with torch.no_grad():
        
        for (data, label_voice_signals,
                        window_idx) in test_loader:
            data, label_voice_signals , window_idx = data.to(device), label_voice_signals.to(device), window_idx.to(device)
            data, means, stds = normalize_input(data)
            valid_length = model.valid_length(data.shape[-1])
            delta = valid_length - data.shape[-1]
            padded = F.pad(data, (delta // 2, delta - delta // 2))

            output_signal = model(padded, window_idx)
            output_signal = center_trim(output_signal, data)

            output_signal = unnormalize_input(output_signal, means, stds)
            output_voices = output_signal[:, 0]

            loss = model.loss(output_voices, label_voice_signals)
            test_loss += loss.item()

            
            #outputs = model(images)
            #_, predicted = torch.max(outputs.data, 1)
            #total += labels.size(0)
            #correct += (predicted == labels).sum().item()

        print(f"loss on test of the model on the {test_loss} ")
        
        wandb.log({"test_loss": test_loss})

    # Save the model in the exchangeable ONNX format
    torch.onnx.export(model, data, "model.onnx")
    wandb.save("model.onnx")



model = model_pipeline(config)