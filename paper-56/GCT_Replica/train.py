import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import GCTModel
from utils import load_data, create_adj_matrix, accuracy, create_dataloader

def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for inputs, adj_matrix, labels in dataloader:
        optimizer.zero_grad()

        outputs = model(inputs, adj_matrix)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_acc += accuracy(outputs, labels) * inputs.size(0)
    return running_loss / len(dataloader.dataset), running_acc / len(dataloader.dataset)

def validate(model, dataloader, criterion):
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
    train_data = load_data('train_data.pkl')
    val_data = load_data('val_data.pkl')

    train_adj_matrix = create_adj_matrix(train_data)
    val_adj_matrix = create_adj_matrix(val_data)

    train_dataloader = create_dataloader(train_data, train_adj_matrix, train_labels, batch_size=16)
    val_dataloader = create_dataloader(val_data, val_adj_matrix, val_labels, batch_size=16)

    in_features = ... # Determine based on input data
    out_features = ... # Determine based on desired output
    d_model = ... # Choose model dimension
    nhead = ... # Choose number of attention heads
    num_layers = ... # Choose number of layers

    model = GCTModel(in_features, out_features, d_model, nhead, num_layers)
    criterion = model.loss_function
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_dataloader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_dataloader, criterion)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

    torch.save(model.state_dict(), 'gct_model.pth')
