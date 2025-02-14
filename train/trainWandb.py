import yaml
import torch.nn as nn
import torch.optim as optim
import torch
import multiprocessing
import os
import torch.nn.functional as F
import tqdm
import wandb
import sys
import soundfile as sf
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from custom_dataset import customdataset
from model import CosNetwork
from model import load_pretrain,center_trim,normalize_input,unnormalize_input

def test_epoch(model: nn.Module, device: torch.device,
               test_loader: torch.utils.data.dataloader.DataLoader,
               log_interval: int = 20) -> float:

    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (data, label_voice_signals,
                        window_idx) in enumerate(test_loader):
            data = data.to(device)
            label_voice_signals = label_voice_signals.to(device)
            window_idx = window_idx.to(device)

            # Normalize input, each batch item separately
            data, means, stds = normalize_input(data)

            valid_length = model.valid_length(data.shape[-1])
            delta = valid_length - data.shape[-1]
            padded = F.pad(data, (delta // 2, delta - delta // 2))

            # Run through the model
            output_signal = model(padded, window_idx)
            output_signal = center_trim(output_signal, data)

            # Un-normalize
            output_signal = unnormalize_input(output_signal, means, stds)
            output_voices = output_signal[:, 0]

            loss = model.loss(output_voices, label_voice_signals)
            test_loss += loss.item()

            if batch_idx % log_interval == 0:
                print("Loss: {}".format(loss))
            
        test_loss /= len(test_loader)
        print("\nTest set: Average Loss: {:.4f}\n".format(test_loss))

        return test_loss
def train_epoch(model: nn.Module, device: torch.device,
                optimizer: optim.Optimizer,
                train_loader: torch.utils.data.dataloader.DataLoader,
                epoch: int, log_interval: int = 20) -> float:
    """
    Train a single epoch.
    """
    # Set the model to training.
    model.train()

    # Training loop
    losses = []
    interval_losses = []
    for batch_idx, (data, label_voice_signals,
                    window_idx) in enumerate(train_loader):
        #print("data , label voice signals, window idx")
        #print(data)
        #print(label_voice_signals)
        #print(window_idx)
        data = data.to(device)
        label_voice_signals = label_voice_signals.to(device)
        window_idx = window_idx.to(device)
        # Normalize input, each batch item separately
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

        loss = model.loss(output_voices, label_voice_signals)

        interval_losses.append(loss.item())
        # Backpropagation
        loss.backward()

        # Gradient clipping
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        # Update the weights
        optimizer.step()

        # Print the loss
        if batch_idx % log_interval == 0:
            log_message = "Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                np.mean(interval_losses))
            
            print(log_message)

            # Open the log file in append mode and write the log message
            with open("log.txt", "a") as log_file:
                log_file.write(log_message + "\n")

            losses.extend(interval_losses)
            interval_losses = []
        del data, label_voice_signals, output_signal
        
    return np.mean(losses)

def get_data(slice=5, train=True):
    full_dataset = torchvision.datasets.MNIST(root=".",
                                              train=train, 
                                              transform=transforms.ToTensor(),
                                              download=True)
    #  equiv to slicing with [::slice] 
    sub_dataset = torch.utils.data.Subset(
      full_dataset, indices=range(0, len(full_dataset), slice))
    
    return sub_dataset


def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True, num_workers=2)
    return loader


def make(config):
    # Make the data
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # Make the model
    model = ConvNet(config.kernels, config.classes).to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)
    
    return model, train_loader, test_loader, criterion, optimizer
def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(project="pytorch-demo", config=hyperparameters):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, train_loader, test_loader, criterion, optimizer = make(config)
      print(model)

      # and use them to train the model
      train(model, train_loader, criterion, optimizer, config)

      # and test its final performance
      test(model, test_loader)

    return model


if __name__ == "__main__":
    




    # Print a starting message
    
    print("Starting the script...")
    
    # Define the path to the YAML file
    yaml_file_path = Path(__file__).resolve().parent.parent / 'constants.yaml'
    print(f"YAML file path: {yaml_file_path}")

    # Load the configuration file
    with open(yaml_file_path, "r") as file:
        config = yaml.safe_load(file)
    print(f"Configuration: {config}")

    # Extract paths from the configuration
    train_dir = config["train_folder_Dataloader"]
    pretrain_path = config["pretrain_path"]
    checkpoints_dir = config["checkpoints_path"]
    learning_rate = config["learning_rate"]
    start_epoch = config["start_epoch"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    name = "multimic_experiment"
    
     
    data_train = customdataset("../gannot-lab-UG/ilya_tomer/OUTPUTS/OUTPUTS_BASIC_TRAIN/ThreeSeconds/",n_mics = 6,sr = 44100,perturb_prob=1.0,mic_radius=0.0725)
    
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    device = torch.device('cuda' if use_cuda else 'cpu')
    num_workers = min(multiprocessing.cpu_count(), 4)
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': True
    } if use_cuda else {}
    print("asd")
    train_loader = DataLoader(data_train, batch_size=8, shuffle=True,**kwargs)
    print("asd2")
    url = 'https://dl.fbaipublicfiles.com/demucs/v3.0/demucs-e07c671f.th'
    state_dict = torch.hub.load_state_dict_from_url(url)
    print("asd3")
    model = CosNetwork()
    load_pretrain(model,state_dict)
    model.to(device)
    print(device)
    if not os.path.exists(os.path.join(checkpoints_dir, name)):
        os.makedirs(os.path.join(checkpoints_dir, name))
        
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    start_epoch = 0
    if start_epoch:
        checkpoints_dir = "checkpoints/cos"
        name = "multimic_experiment"
        checkpoints_dir = os.path.join(checkpoints_dir,name)
        print("hola")
        checkpoint_path = os.path.join(checkpoints_dir, "{}_6layer_10000_exact_finetune.pt".format(int(start_epoch) - 1))
        state_dict = torch.load(checkpoint_path)
        load_pretrain(model, state_dict)
    else:
        start_epoch = 0

    train_losses = []

    try:
        for epoch in range(start_epoch, start_epoch + epochs + 1):
            train_loss = train_epoch(model, device, optimizer, train_loader, epoch)
            train_losses.append(train_loss)
            if epoch % 2 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(checkpoints_dir,name,"{}_6layer_10000_exact_finetune.pt".format(epoch))
                )
            print(f"train_loss: {train_losses}")
            plt.plot(train_losses)
            plt.savefig(f"train_losses{epoch}.png")
    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as _:
        import traceback
        traceback.print_exc()