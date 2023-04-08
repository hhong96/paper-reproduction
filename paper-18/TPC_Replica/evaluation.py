import torch
from torch.utils.data import DataLoader, TensorDataset
from model import TPCModel
from utils import load_data, create_dataloader

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

if __name__ == '__main__':
    test_data = load_data('test_data.pkl')
    test_dataloader = create_dataloader(test_data, test_labels, batch_size=16)

    input_size = ... # Determine based on input data
    hidden_size = ... # Choose hidden layer size
    output_size = ... # Determine based on desired output
    num_layers = ... # Choose number of layers

    model = TPCModel(input_size, hidden_size, output_size, num_layers=num_layers)
    model.load_state_dict(torch.load('tpc_model.pth'))
    criterion = model.loss_function

    test_loss = evaluate(model, test_dataloader, criterion)
    print(f'Test Loss: {test_loss:.4f}')
