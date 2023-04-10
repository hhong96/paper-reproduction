import torch
from torch.utils.data import DataLoader, TensorDataset
from model import GCTModel
from utils import load_data, create_adj_matrix, accuracy, create_dataloader

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for inputs, adj_matrix, labels in dataloader:
            outputs = model(inputs, adj_matrix)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_acc += accuracy(outputs, labels) * inputs.size(0)
    return running_loss / len(dataloader.dataset), running_acc / len(dataloader.dataset)

if __name__ == '__main__':
    test_data = load_data('test_data.pkl')
    test_adj_matrix = create_adj_matrix(test_data)
    test_dataloader = create_dataloader(test_data, test_adj_matrix, test_labels, batch_size=16)

    in_features = ... # Determine based on input data
    out_features = ... # Determine based on desired output
    d_model = ... # Choose model dimension
    nhead = ... # Choose number of attention heads
    num_layers = ... # Choose number of layers

    model = GCTModel(in_features, out_features, d_model, nhead, num_layers)
    model.load_state_dict(torch.load('gct_model.pth'))
    criterion = model.loss_function

    test_loss, test_acc = evaluate(model, test_dataloader, criterion)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
