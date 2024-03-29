import argparse
import json
import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from model import BinaryClassifier

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # Load the parameters used to create the model
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BinaryClassifier(model_info['input_features'], model_info['hidden_dim'], model_info['output_dim'])

    # Load the stored model parameters
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Set to eval mode, could use no_grad
    model.to(device).eval()

    print("Done loading model.")
    return model

# Gets training data in batches from the train.csv file
def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")

    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)
    
    # Labels are in the first column
    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    # Features are in the remaining columns
    train_x = torch.from_numpy(train_data.drop([0], axis=1).values).float()

    # Create the dataset
    train_ds = torch.utils.data.TensorDataset(train_x, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


# Provided training function
def train(model, train_loader, epochs, criterion, optimizer, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model to be trained.
    train_loader - The PyTorch DataLoader used during training.
    epochs       - The total number of epochs to train for.
    criterion    - The loss function used for training. 
    optimizer    - The optimizer to use during training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train() # Make sure that the model is in training mode.

        total_loss = 0

        for batch in train_loader:
            # Get data
            batch_x, batch_y = batch

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # Get predictions from model
            y_pred = model(batch_x)
            
            # Perform backprop
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data.item()

        print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(train_loader)))


if __name__ == '__main__':
    
    # All of the model paramaters and training parameters are sent as arguments
    # when this script is executed during a training job.

    # Set up argument parser to access parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data/saving models; automatically set
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    
    # Model Parameters
    # Args for model parameters: input_features, hidden_dim, output_dim
    parser.add_argument('--input_features', type=int, default=2, metavar='IN',
                        help='number of input features to model (default: 2)')
    parser.add_argument('--hidden_dim', type=int, default=10, metavar='H',
                        help='hidden dim of model (default: 10)')
    parser.add_argument('--output_dim', type=int, default=1, metavar='OUT',
                        help='output dim of model (default: 1)')
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    # Set seed for generating random numbers
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load the training data
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)

    # Build the model with input params.
    # Model moved to designated device (gpu/cpu).
    model = BinaryClassifier(args.input_features, args.hidden_dim, args.output_dim).to(device)

    # Define an optimizer and loss function for training
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # Train the model
    train(model, train_loader, args.epochs, criterion, optimizer, device)

    # Save model params
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_features': args.input_features,
            'hidden_dim': args.hidden_dim,
            'output_dim': args.output_dim,
        }
        torch.save(model_info, f)

	# Save the model 
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)