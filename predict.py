import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image



class IBD_CNN(nn.Module):
    def __init__(self):
        super(IBD_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 54 * 54, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x




model = IBD_CNN()
model.load_state_dict(torch.load("ibd_cnn_model.pth"))
model.eval()

transform = transforms.Compose([


    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])


image_path = "test_image.jpg"   # change if needed
image = Image.open(image_path).convert("RGB")
image = transform(image)
image = image.unsqueeze(0)



with torch.no_grad():
    output = model(image)
    print("UC probability:", output.item())
    prediction = (output >= 0.5).item()

if prediction == 1:
    print("Prediction: Ulcerative Colitis")
else:
    print("Prediction: Normal")
