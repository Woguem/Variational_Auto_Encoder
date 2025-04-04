"""
@author: Dr Yen Fred WOGUEM 

@description: This script trains a Variational AE model to generate image

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

start_time = datetime.now()  # Start timer


# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
latent_dim = 20
lr = 0.001
num_epochs = 20

# Data processing
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Variational AE
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_dim)  # Moyenne
        self.fc22 = nn.Linear(400, latent_dim)  # Log variance
        
        # Décodeur
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)
    
    def encode(self, x):
        h1 = F.relu(self.fc1(x.view(-1, 784)))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Initialization
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training
for epoch in range(num_epochs):
    total_loss = 0
    for data in train_loader:
        img, _ = data
        img = img.to(device)
        
        # Forward pass
        recon_img, mu, logvar = model(img)
        loss = loss_function(recon_img, img, mu, logvar)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader.dataset):.4f}')

    # Visualization
    if epoch % 3 == 0:
        with torch.no_grad():
            sample = torch.randn(1, latent_dim).to(device)
            generated = model.decode(sample).cpu()
            plt.imshow(generated.view(28,28).numpy(), cmap='gray')
            #plt.title('Image générée')
            plt.savefig(f'Generated_Image_{epoch}.png') 
            plt.close()




end_time = datetime.now()  # End of timer
execution_time = end_time - start_time
print(f"\nDurée d'exécution : {execution_time}")
