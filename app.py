from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import webbrowser

app = Flask(__name__)

# -------------------------------
# Model Architecture (same as train)
# -------------------------------

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

# -------------------------------
# Load Model
# -------------------------------

model = IBD_CNN()
model.load_state_dict(torch.load("ibd_cnn_model.pth", map_location=torch.device('cpu')))
model.eval()

# -------------------------------
# Image Transform (same as predict)
# -------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# -------------------------------
# Upload folder
# -------------------------------

UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    img_path = None
    probability = None
    accuracy = 0.90
    precision = 0.87
    recall = 0.94

    if request.method == "POST":
        file = request.files["image"]

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            image = Image.open(filepath).convert("RGB")
            image = transform(image)
            image = image.unsqueeze(0)

            with torch.no_grad():
                output = model(image)
                probability = output.item()
                pred = (output >= 0.5).item()

            if pred == 1:
                prediction = "Ulcerative Colitis"
            else:
                prediction = "Normal"

            img_path = filepath

            accuracy = 0.90
            precision = 0.87
            recall = 0.94

    return render_template("index.html",
                       prediction=prediction,
                       img_path=img_path,
                       probability=probability,
                       accuracy=accuracy,
                       precision=precision,
                       recall=recall)


if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)
