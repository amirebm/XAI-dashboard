# model.py
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

# Define Simple CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to load the saved model
def load_model():
    model = SimpleCNN()
    model.load_state_dict(torch.load("simple_cnn.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Function to get integrated gradients
def get_integrated_gradients(model, image, target_label):
    baseline = torch.zeros_like(image)
    integrated_gradients = IntegratedGradients(model)
    attributions, _ = integrated_gradients.attribute(image, baseline, target=target_label, return_convergence_delta=True)
    return attributions.squeeze().detach().numpy()
