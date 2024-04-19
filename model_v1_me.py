import torch
import torch.nn as nn

class ProteinModel(nn.Module):
    def __init__(self):
        super(ProteinModel, self).__init__()
        self.fc1 = nn.Linear(1280, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)

    def forward(self, data: dict):
        """
        :param data: Dictionary containing ['embedding', 'mutant', 'mutant_sequence',
                                                'logits', 'wt_logits', 'wt_embedding']
        :return: predicted DMS score
        """
        # Get the embeddings
        x = data['embedding']-data['wt_embedding']
        # Move x to gpu
        if torch.cuda.is_available():
            device = "cuda" # Use NVIDIA GPU (if available)
        elif torch.backends.mps.is_available():
            device = "mps" # Use Apple Silicon GPU (if available)
        else:
            device = "cpu" # Default to CPU if no GPU is available
        #print(f"Using device: {device}")      
        
        x = x.to(device)

        x = self.fc1(x)
        x = self.relu(x)
        #print(x.shape)
        x = torch.sum(x, dim=1)
        x = self.fc2(x)
        return x