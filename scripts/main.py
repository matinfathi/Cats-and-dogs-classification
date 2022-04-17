from torch_model import *

if __name__ == '__main__':

    # Hyperparameters and constants
    DATASET_PATH = '<PATH_TO_THE_DATASET>'
    SAVE_PATH = '<PATH_FOR_SAVING_THE_MODEL_WEIGHTS>'
    CLASSES = {'Cat': 0, 'Dog': 1}
    BATCH_SIZE = 32
    IMAGE_SIZE = 50
    SIZE_DATA = None
    EPOCHS = 50
    LR = 0.001

    # Specify using GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Store dataset in list
    dataset = store_data(CLASSES, DATASET_PATH, SIZE_DATA, IMAGE_SIZE)

    # Create Dataloaders of train and test data
    train_dataloader, test_dataloader = create_dataloader(dataset, BATCH_SIZE)

    # Define Convolution layer and move it to defined device
    model = ConvNet().to(device)

    # Define the loss function and optimizer for the model
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Train the model with train data
    train(train_dataloader, EPOCHS, device, model, criterion, optimizer, SAVE_PATH)

    # Test the model with test data
    evaluation(test_dataloader, device, model)
