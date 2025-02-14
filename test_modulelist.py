import torch
import torch.nn as nn
class SequentialModel(nn.Module):
    def __init__(self):
        super(SequentialModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 20),  # Input layer
            nn.ReLU(),
            nn.Linear(20, 15),  # Hidden layer
            nn.ReLU(),
            nn.Linear(15, 1)    # Output layer
        )
        
    def forward(self, x):
        return self.model(x)
class ModuleListModel(nn.Module):
    def __init__(self):
        super(ModuleListModel, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(10, 20),  # Input layer
            nn.ReLU(),
            nn.Linear(20, 15),  # Hidden layer
            nn.ReLU(),
            nn.Linear(15, 1)    # Output layer
        ])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
# Create sample input data: batch size of 5, input features of size 10
input_data = torch.randn(5, 10)

# Instantiate the models
sequential_model = SequentialModel()
modulelist_model = ModuleListModel()

# Pass the input data through both models
output_sequential = sequential_model(input_data)
output_modulelist = modulelist_model(input_data)

# Print the outputs
print("Output from Sequential Model:\n", output_sequential)
print("\nOutput from ModuleList Model:\n", output_modulelist)
