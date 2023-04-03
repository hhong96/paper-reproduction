import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import TPCModel
from utils import load_data, create_dataloader

def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

if __name__ == '__main__':
    train_data = load_data('train_data.pkl')
    val_data = load_data('val_data.pkl')

    train_dataloader = create_dataloader(train_data, train_labels, batch_size=16)
    val_dataloader = create_dataloader(val_data, val_labels, batch_size=16)

    input_size = ... # Determine based on input data
    hidden_size = ... # Choose hidden layer size
    output_size = ... # Determine based on desired output
    num_layers = ... # Choose number of layers

    model = TPCModel(input_size, hidden_size, output_size, num_layers=num_layers)
    criterion = model.loss_function
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer)
        val_loss = validate(model, val_dataloader, criterion)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')

    torch.save(model.state_dict(), 'tpc_model.pth')
