from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch
import cv2
import os


class ConvNet(torch.nn.Module):
    """
    Convolution Model
    """
    def __init__(self):
        super(ConvNet, self).__init__()
        # First convolution layer
        self.Layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(p=0.1),
        )
        # Second Convolution layer
        self.Layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Dropout(p=0.1),
            torch.nn.Flatten(),
        )
        # Flatten the output of Convolution layers and
        # Feed it to a fully connected layer
        self.Layer3 = torch.nn.Sequential(
            torch.nn.Linear(32*12*12, 128),
            torch.nn.ReLU(),
        )
        # Second fully connected layer with sigmoid function
        self.Layer4 = torch.nn.Sequential(
            torch.nn.Linear(128, 2),
            torch.nn.Sigmoid(),
        )

    def forward(self, batch):
        x = self.Layer1(batch)  # Convolution 1
        x = self.Layer2(x)      # Convolution 2
        x = self.Layer3(x)      # Linear 1
        x = self.Layer4(x)      # Linear 2

        return x


def create_dataloader(dataset, batch_size=32):
    """
    Create and return pytorch dataloader form data in a list.
    """
    # Split train and test
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, lengths=[train_size, test_size])

    # Create dataloader from train and test
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader


def store_data(classes, dataset_path, size_data, image_size):
    """
    Extract data from given path and store it in a list.
    """
    dataset = []

    # Iterate over classes and collect data from dataset folder
    for c in classes.keys():
        class_path = os.path.join(dataset_path, c)  # find path to each class
        list_class = os.listdir(class_path)  # make a list of all files
        total = size_data if size_data else len(list_class)  # use all files or a part of them

        # Iterate over images
        for image_name in tqdm(list_class[:total], total=total):
            if image_name.endswith('.jpg'):
                item_path = os.path.join(class_path, image_name)  # find the path to each image
                try:
                    # Read image and convert it to grayscale
                    image = cv2.imread(item_path, cv2.IMREAD_GRAYSCALE)
                    # Resize image to a smaller shape
                    image_resized = cv2.resize(image, (image_size, image_size)).reshape(1, image_size, image_size)
                    # Append it to a list
                    dataset.append((image_resized, classes[c]))
                except Exception as e:
                    pass

    return dataset


def train(train_dataloader, epochs, device, model, criterion, optimizer, save_path):
    """
    Train our model with specified data.
    """
    total_step = len(train_dataloader)
    accuaracies, losses = [], []

    # Iterate over number of epochs
    for epoch in range(epochs):
        correct = 0
        total = 0

        # Iterate over each batch
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.float().to(device)  # convert imput data to float
            labels = labels.to(device)

            # feed input data to model
            outputs = model(images)
            # calculate loss
            loss = criterion(outputs, labels)

            # set gradients to zero
            optimizer.zero_grad()
            # perform backpropragation
            loss.backward()
            # update parameters
            optimizer.step()

            # find correct matches
            correct += (torch.argmax(outputs, dim=1) == labels).float().sum()
            total += labels.size(0)

        # calculate accuracy for each epoch
        accuracy = (100 * correct) / total
        accuaracies.append(accuracy)
        losses.append(loss)

        # print the information
        if (epoch + 1) % 10 == 0:
            print(
                f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{total_step}], Loss: {round(loss.item(), 4)}'
                f', Accuracy: {accuracy} '
            )
        print()

    torch.save(model.state_dict(), save_path)


def evaluation(test_dataloader, device, model):
    """
    Test the accuracy of the model with test data.
    """
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():  # we want no gradients for evaluation
        correct = 0
        total = 0
        # iterate over test data
        for images, labels in test_dataloader:
            images = images.float().to(device)
            labels = labels.to(device)
            # feed data to model
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Test Accuracy of the model on the test images: %{round(accuracy, 4)}')
