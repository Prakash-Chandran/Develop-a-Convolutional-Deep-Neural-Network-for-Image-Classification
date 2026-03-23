# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Image classification is a fundamental task in computer vision, where the goal is to assign an input image to one of several predefined categories. Traditional machine learning approaches often struggle to capture complex patterns in images. Therefore, this project aims to develop a Convolutional Deep Neural Network (CNN) that can automatically learn hierarchical features from images and perform accurate classification.

The objective is to design and implement a CNN model using multiple layers such as convolutional layers, pooling layers, and fully connected layers to extract and learn important features from the dataset. The model should be trained and evaluated on a labeled image dataset to classify images into their respective classes.

The developed model should achieve high accuracy, effectively generalize to unseen data, and demonstrate the advantages of deep learning techniques in image classification tasks.

## Neural Network Model

<img width="1300" height="657" alt="567637318-0e9e33bc-e53a-4b8a-bd29-38b7bff06d68" src="https://github.com/user-attachments/assets/26a0badc-b171-4b91-8762-59a0643e06bd" />


## DESIGN STEPS
### STEP 1: 

Import the required libraries (torch, torchvision, torch.nn, torch.optim) and load the image dataset with necessary preprocessing like normalization and transformation.
### STEP 2: 

Split the dataset into training and testing sets and create DataLoader objects to feed images in batches to the CNN model.

### STEP 3: 

Define the CNN architecture using convolutional layers, ReLU activation, max pooling layers, and fully connected layers as implemented in the CNNClassifier class.

### STEP 4: 

Initialize the model, define the loss function (CrossEntropyLoss), and choose the optimizer (Adam) for training the network.

### STEP 5: 

Train the model using the training dataset by performing forward pass, computing loss, backpropagation, and updating weights for multiple epochs.

### STEP 6: 
Evaluate the trained model on test images and verify the classification accuracy for new unseen images.




## PROGRAM

### Name: PRAKASH C

### Register Number: 212223240122

```python
class CNNClassifier(nn.Module):
    def __init__(self, input_size):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(128*3*3,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x



# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Model
def train_model(model, train_loader, num_epochs=3):

    for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()e

        
        
        
        print('Name: PRAKASH C')
        print('Register Number:212223240122')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

### OUTPUT

## Training Loss per Epoch

<img width="506" height="342" alt="image" src="https://github.com/user-attachments/assets/7fc3df83-3524-4443-a26e-317b9f4c506a" />


## Confusion Matrix

<img width="860" height="697" alt="image" src="https://github.com/user-attachments/assets/c845d2a1-30b6-4ce3-af9a-d4f67bd6ba51" />


## Classification Report

<img width="541" height="346" alt="image" src="https://github.com/user-attachments/assets/95927f25-17ae-4308-b9ff-5f827336115e" />

### New Sample Data Prediction

<img width="545" height="500" alt="image" src="https://github.com/user-attachments/assets/ff47ddea-0578-4045-bf0f-c03f0a0488ba" />


## RESULT
The program Exucuted Successfully.
