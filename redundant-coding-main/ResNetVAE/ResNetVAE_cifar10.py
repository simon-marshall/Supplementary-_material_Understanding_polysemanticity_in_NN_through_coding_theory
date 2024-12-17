import os
import wandb
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from modules import *

dropouts = [0.0, 0.2, 0.4, 0.6, 0.8] #  
for dropout_p in dropouts:

    # EncoderCNN architecture
    CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 1024
    CNN_embed_dim = 256     # latent dim extracted by 2D CNN
    res_size = 224        # ResNet image size
    # dropout_p = 0.2       # dropout probability


    # training parameters
    epochs = 200 # training epochs
    batch_size = 50
    learning_rate = 1e-3
    log_interval = 10 # interval for displaying training info

    args = {
        "CNN_fc_hidden1": CNN_fc_hidden1,
        "CNN_fc_hidden2": CNN_fc_hidden2,
        "res_size": res_size,
        "dropout_p": dropout_p,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "log_interval": log_interval,
    }

    wandb.init(config=args, project="jkirchner-sandbox", name=f"ResNetVAE_cifar10_dropout_{dropout_p}")

    # save model
    save_model_path = './results_cifar10'

    def check_mkdir(dir_name):
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)


    def loss_function(recon_x, x, mu, logvar):
        # MSE = F.mse_loss(recon_x, x, reduction='sum')
        MSE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD

    def train(log_interval, model, device, train_loader, optimizer, epoch):
        # set model as training mode
        model.train()

        losses = []
        all_y, all_z, all_mu, all_logvar = [], [], [], []
        N_count = 0   # counting total trained sample in one epoch
        for batch_idx, (X, y) in enumerate(train_loader):
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )
            N_count += X.size(0)

            optimizer.zero_grad()
            X_reconst, z, mu, logvar = model(X)  # VAE
            loss = loss_function(X_reconst, X, mu, logvar)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

            all_y.extend(y.data.cpu().numpy())
            all_z.extend(z.data.cpu().numpy())
            all_mu.extend(mu.data.cpu().numpy())
            all_logvar.extend(logvar.data.cpu().numpy())

            # show information
            if (batch_idx + 1) % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item()))

        all_y = np.stack(all_y, axis=0)
        all_z = np.stack(all_z, axis=0)
        all_mu = np.stack(all_mu, axis=0)
        all_logvar = np.stack(all_logvar, axis=0)

        # save Pytorch models of best record
        torch.save(model.state_dict(), os.path.join(save_model_path, 'model_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
        torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
        print("Epoch {} model saved!".format(epoch + 1))

        return X_reconst.data.cpu().numpy(), all_y, all_z, all_mu, all_logvar, losses

    def validation(model, device, optimizer, test_loader):
        # set model as testing mode
        model.eval()

        test_loss = 0
        all_y, all_z, all_mu, all_logvar = [], [], [], []
        with torch.no_grad():
            for X, y in test_loader:
                # distribute data to device
                X, y = X.to(device), y.to(device).view(-1, )
                X_reconst, z, mu, logvar = model(X)

                loss = loss_function(X_reconst, X, mu, logvar)
                test_loss += loss.item()  # sum up batch loss

                all_y.extend(y.data.cpu().numpy())
                all_z.extend(z.data.cpu().numpy())
                all_mu.extend(mu.data.cpu().numpy())
                all_logvar.extend(logvar.data.cpu().numpy())

        test_loss /= len(test_loader.dataset)
        all_y = np.stack(all_y, axis=0)
        all_z = np.stack(all_z, axis=0)
        all_mu = np.stack(all_mu, axis=0)
        all_logvar = np.stack(all_logvar, axis=0)

        # show information
        print('\nTest set ({:d} samples): Average loss: {:.4f}\n'.format(len(test_loader.dataset), test_loss))
        return X_reconst.data.cpu().numpy(), all_y, all_z, all_mu, all_logvar, test_loss


    # Detect devices
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

    # Data loading parameters
    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])])

    # cifar10 dataset (images and labels)
    cifar10_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar10_test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Data loader (input pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=cifar10_train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=cifar10_test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    resnet_vae = ResNet_VAE(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)

    print("Using", torch.cuda.device_count(), "GPU!")
    model_params = list(resnet_vae.parameters())
    optimizer = torch.optim.Adam(model_params, lr=learning_rate)

    # record training process
    epoch_train_losses = []
    epoch_test_losses = []
    check_mkdir(save_model_path)

    # start training
    for epoch in range(epochs):
        # train, test model
        X_reconst_train, y_train, z_train, mu_train, logvar_train, train_losses = train(log_interval, resnet_vae, device, train_loader, optimizer, epoch)
        X_reconst_test, y_test, z_test, mu_test, logvar_test, epoch_test_loss = validation(resnet_vae, device, optimizer, valid_loader)

        wandb.log({"test_loss": epoch_test_loss, "epoch": epoch})
        wandb.log({"train_loss": np.mean(train_losses), "epoch": epoch})

        # save results
        epoch_train_losses.append(train_losses)
        epoch_test_losses.append(epoch_test_loss)

        # save all train test results
        A = np.array(epoch_train_losses)
        C = np.array(epoch_test_losses)
        
        np.save(os.path.join(save_model_path, 'ResNet_VAE_training_loss.npy'), A)
        np.save(os.path.join(save_model_path, 'y_cifar10_train_epoch{}.npy'.format(epoch + 1)), y_train)
        np.save(os.path.join(save_model_path, 'z_cifar10_train_epoch{}.npy'.format(epoch + 1)), z_train)

        torch.save(resnet_vae.state_dict(), f"resnet_vae_{dropout_p}_{epoch}.pth")
        
    wandb.finish()