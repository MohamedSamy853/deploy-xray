import numpy as np
import torch
from torch import nn
from torchvision import models , transforms
from PIL import Image
from flask import Flask , request , jsonify
import subprocess
import gdown
url = 'https://drive.google.com/file/d/1EzFM9aKDidqJZMu4y15ohM22IzVPxwRd/view?usp=sharing'
output = 'model_state.pth'
if not os.path.isfile(output):
     subprocess.run(['python','-m','pip', 'install', '--upgrade', '--no-cache-dir', 'gdown'])
     gdown.download(url, output, quiet=False, use_cookies=False , fuzzy=True)
     
index2label ={0: 'Infiltration',
 1: 'Atelectasis',
 2: 'Effusion',
 3: 'Nodule',
 4: 'Mass',
 5: 'Pneumothorax',
 6: 'Consolidation',
 7: 'Pleural_Thickening',
 8: 'Cardiomegaly',
 9: 'Edema'}
transform = transforms.Compose([
    transforms.Lambda(lambda x:x/255.0),
    transforms.Resize((200, 250),antialias=False),
    
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
model_resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
class XRayModel(nn.Module):
    def __init__(self,model=model_resnet, output_class=10):
        super().__init__()
        self.model = model
       
        self.output_class = output_class
        self.avgpool = nn.Sequential(
            nn.Conv2d(2048 , 1024 , 3, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(1024 , 1024 , 3, 1),
            nn.Conv2d(1024 , 1024 , 3, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024),
            nn.AdaptiveAvgPool2d(output_size = 1)
            )
        self.fc =nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512 , self.output_class),
            )
        self.model.avgpool = self.avgpool

        self.model.fc = self.fc
       # self.tfs = tfs
    def forward(self, x):
        x = self.model(x)
        
        return x 
model = XRayModel().to('cpu')
def load_model(path=output):
    state_dict = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    return model.eval()
model = load_model()
 
def predict_image(image):
    img  : torch.Tensor= transform(image)
    img = img[None]
    res = model(img)
    res = res.cpu()
    res = torch.softmax(res, dim=1)
    top3_prob = torch.topk(res, k=3, dim=1)
    top3_indices = top3_prob.indices
    print(top3_indices , top3_prob.values)
    return top3_indices.int() , top3_prob.values
app = Flask(__name__)

@app.route("/")
def home():
    return "hello from home page"
@app.route("/predict", methods=['POST', 'GET'])
def predict():
    file = request.files['image']
    image = Image.open(file) 
    image = np.array(image)
    if image.ndim !=3:
        image = image[... ,np.newaxis]
        image = image.repeat(3, 2)
    image = torch.from_numpy(image).float()
    image = image.permute(2 , 0, 1)
    indices , prob = predict_image(image)
    res = {index2label[i.item()]:p.item().__round__(3) for i, p in zip(indices.flatten() , prob.flatten())}
    return jsonify({"pred":res})
    

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
