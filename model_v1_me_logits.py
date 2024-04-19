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

        # Get logits of the mutation:
        def get_mutated_position_idx(m):
            return int(m[1:-1])
        
        # Get logits for mut and wt for each mutant
        mt_logit_tensor = torch.tensor([])
        wt_logit_tensor = torch.tensor([])
        for i in range(x.shape[0]):
            #logit_vector = torch.cat((data['logits'][i,get_mutated_position_idx(data['mutant'][i]),:],data['wt_logits'][i,get_mutated_position_idx(data['mutant'][i]),:]))
            mt_logit_vector = torch.cat((data['logits'][i,get_mutated_position_idx(data['mutant'][i]),:],torch.zeros(1280-33)))
            wt_logit_vector = torch.cat((data['wt_logits'][i,get_mutated_position_idx(data['mutant'][i]),:],torch.zeros(1280-33)))
            mt_logit_tensor = torch.cat((mt_logit_tensor, mt_logit_vector.unsqueeze(0)), dim=0)
            wt_logit_tensor = torch.cat((wt_logit_tensor, wt_logit_vector.unsqueeze(0)), dim=0)
        
        # Add the logit tensors to the input tensor
        mt_logit_tensor = mt_logit_tensor.unsqueeze(1)
        wt_logit_tensor = wt_logit_tensor.unsqueeze(1)

        x = torch.hstack((x,mt_logit_tensor))
        x = torch.hstack((x,wt_logit_tensor))

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