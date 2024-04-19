import torch
import torch.nn as nn
import re

class ProteinModel(nn.Module):
    def __init__(self):
        super(ProteinModel, self).__init__()
        self.fc1 = nn.Linear(1280, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        #self.fc3 = nn.Linear(128, 1)

    def forward(self, data: dict):
        """
        :param data: Dictionary containing ['embedding', 'mutant', 'mutant_sequence',
                                                'logits', 'wt_logits', 'wt_embedding']
        :return: predicted DMS score
        """
        # Get the embeddings
        x = data['embedding']-data['wt_embedding']

        def get_mutated_position_idx(m):
            return int(m[1:-1])

        mut_x = []
        for i in range(x.shape[0]):
            #print(i)
            mut_x.append(x[i,get_mutated_position_idx(data['mutant'][i]),:])
        
        mut_x = torch.stack(mut_x)

        # Move x to gpu
        if torch.cuda.is_available():
            device = "cuda" # Use NVIDIA GPU (if available)
        elif torch.backends.mps.is_available():
            device = "mps" # Use Apple Silicon GPU (if available)
        else:
            device = "cpu" # Default to CPU if no GPU is available
        #print(f"Using device: {device}")      
        
        mut_x = torch.unsqueeze(mut_x, 1)
        #print(mut_x.shape)
        mut_x = mut_x.to(device)

        mut_x = self.fc1(mut_x)
        mut_x = self.relu(mut_x)
        mut_x = torch.sum(mut_x, dim=1)
        mut_x = self.fc2(mut_x)
        return mut_x