import os.path
import sys
import warnings
import time
import traceback
import pdb

# supress pandas deprication warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import torch
import torch.optim as optim
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
try:
    from src.data_loader import get_dataloader
    from src.model_v1_me import ProteinModel
except:
    from data_loader import get_dataloader
    from model_v1_me import ProteinModel

torch.manual_seed(0)

DATA_DIR = 'data'
OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_spearman = float('-inf')
        self.epoch_counter = 0
        #print(self.max_spearman,self.min_delta)

    def early_stop(self, val_spearman):
        #print(self.max_spearman,self.min_delta)
        if self.epoch_counter < 5:
            self.epoch_counter += 1
            return False
        else:
            if val_spearman > self.max_spearman:
                self.max_spearman = val_spearman
                self.counter = 0
            elif val_spearman < (self.max_spearman + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
        return False

def plot_metrics(metrics, title='Training Metrics'):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i, (k, v) in enumerate(metrics[0].items()):
        ax = axs[i // 2, i % 2]
        ax.plot([m[k] for m in metrics])
        ax.set_title(k)
    # add sup title
    fig.suptitle(title)
    save_path = f'{OUT_DIR}/{title}.png'
    plt.savefig(save_path)
    print(f"Saved plot of to {os.getcwd()}/{save_path}")
    plt.show()

def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for data in test_loader:
            labels = data['DMS_score']
            outputs = model(data)
            outputs = outputs.to('cpu')
            predictions.extend(outputs.numpy())
            actuals.extend(labels.numpy())
    metrics = get_performance_metrics(predictions, actuals)
    return metrics

def get_performance_metrics(predictions, actuals):
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    spearman = spearmanr(actuals, predictions)
    return {'mse': mse, 'mae': mae, 'r2': r2, 'spearman_r': spearman.correlation}

def train_model(model, train_loader, validation_loader, test_loader , plot=False, data_name = 'HUMAN',model_name='model'):   
    
    # Define training parameters
    num_epochs = 100
    early_stopper = EarlyStopper(patience=5, min_delta=0.02) 
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Defining the lists to store the metrics
    train_epoch_metrics = []
    val_epoch_metrics = []
    test_epoch_metrics = []

    # GPU check
    if torch.cuda.is_available():
        device = "cuda" # Use NVIDIA GPU (if available)
    elif torch.backends.mps.is_available():
        device = "mps" # Use Apple Silicon GPU (if available)
    else:
        device = "cpu" # Default to CPU if no GPU is available
    print(f"Using device: {device}")
    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        # Putting model in training mode and setting params
        model.train()
        running_loss = 0.0
        pred_vals = []
        true_vals = []
        for i, data in enumerate(train_loader):
            # Preparing data and labels
            labels = data['DMS_score']
            optimizer.zero_grad()

            # Train the Model
            #input = data['embedding']
            #print(type(input))
            #input = input.to(device)
            #print(input.device)
            #print(next(model.parameters()).device)
            #print(input.device)

            # Put input through model
            outputs = model(data).squeeze()    
            outputs = outputs.to('cpu')

            # Calculate Metrics
            pred_vals.extend(outputs.detach().numpy())
            true_vals.extend(labels.numpy())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_metrics = get_performance_metrics(pred_vals, true_vals)
        train_epoch_metrics.append(train_metrics)
        epoch_loss = running_loss / len(train_loader)
        val_metrics = evaluate_model(model, validation_loader)
        test_metrics = evaluate_model(model, test_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, MAE: {train_metrics["mae"]:.4f}, '
          f'R2: {train_metrics["r2"]:.4f}, Spearman R: {train_metrics["spearman_r"]:.4f},|| Val Spearman R: {val_metrics["spearman_r"]:.4f}, Val MAE: {val_metrics["mae"]:.4f}, Val R2: {val_metrics["r2"]:.4f} || Test Spearman R: {test_metrics["spearman_r"]:.4f}')
        val_epoch_metrics.append(val_metrics)
        test_epoch_metrics.append(test_metrics)
        if early_stopper.early_stop(val_metrics["spearman_r"]):             
            print(early_stopper.early_stop(test_metrics["spearman_r"]))
            break

    # Save the training and validation epoch metrics
    val_df = pd.DataFrame(val_epoch_metrics)
    train_df = pd.DataFrame(train_epoch_metrics)
    test_df = pd.DataFrame(test_epoch_metrics)
    print(val_df)
    val_df.to_csv(f'{OUT_DIR}/val_metrics_{model_name}_{data_name}.csv', index=False)
    train_df.to_csv(f'{OUT_DIR}/train_metrics_{model_name}_{data_name}.csv', index=False)
    test_df.to_csv(f'{OUT_DIR}/test_metrics_{model_name}_{data_name}.csv', index=False)

    if plot:
        plot_metrics(train_epoch_metrics, title='Training Metrics')
        plot_metrics(val_epoch_metrics, title='Validation Metrics')


def main(experiment_path, train_folds=[1,2,3], validation_folds=[4], test_folds=[5], plot=False):
    print(f"\nTraining model on {experiment_path}")
    train_loader = get_dataloader(experiment_path=experiment_path, folds=train_folds, return_logits=True, return_wt=True)
    val_loader = get_dataloader(experiment_path=experiment_path, folds=validation_folds, return_logits=True)
    test_loader = get_dataloader(experiment_path=experiment_path, folds=test_folds, return_logits=True)
    print('Train loader:',len(train_loader))
    print('Val loader:',len(val_loader))
    print('Test loader:',len(test_loader))
    model = ProteinModel()

    start = time.time()
    train_model(model, train_loader, val_loader, test_loader,plot=plot, data_name = experiment_path.split('/')[-1],model_name=MODEL_NAME)
    train_time = time.time() - start
    metrics = evaluate_model(model, test_loader)
    metrics['train_time_secs'] = round(train_time, 1)
    print("Test performance metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    metrics['DMS_id'] = experiment_path.split('/')[-1]
    return metrics

if __name__ == "__main__":
    MODEL_NAME='model_v1_me'
    start = time.time()
    rows = []
    # if command line argument provided, use that as the experiment name
    # otherwise, loop over all experiments
    if len(sys.argv) > 1:
        experiments = sys.argv[1].split(',')
    else:
        experiments = list(set([fname.split('.')[0] for fname in os.listdir(DATA_DIR) if not fname.startswith('.')]))
    for experiment in experiments:
        try:
            experiment_path = f"{DATA_DIR}/{experiment}"
            new_row = main(experiment_path=experiment_path)
            rows.append(new_row)
        except Exception as e:
            print(f"Error with {experiment}: {e}")
            traceback.print_exc()
            pdb.post_mortem(sys.exc_info()[2])

    df = pd.DataFrame(rows)
    df_savepath = f'{OUT_DIR}/{MODEL_NAME}_{experiment}_results.csv'
    df.to_csv(df_savepath, index=False)
    print(f"Metrics for {len(df)} experiments saved to {os.getcwd()}/{df_savepath}")
    print(df.head())
    end = time.time()
    print(f"Total time: {(end-start)/60:.2f} minutes")