from model import SymbolCNN
import torch
import os
from symbols import symbols
from dataset import image_size, transform
from PIL import Image

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "augmented_models_4", "checkpoint_10.pth")

model = SymbolCNN(num_classes=len(symbols), image_size=image_size)  # Initialize the model
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()  # Set the model to evaluation mode

# img = Image.new("L", (500, 300), 255)
x = torch.randn(1, 1, image_size[0], image_size[1])
# x = transform(img)
# x = x.unsqueeze(0)
print(x.shape)
width = torch.tensor([150])
height = torch.tensor([333])

onnx_program = torch.onnx.dynamo_export(model, x, width, height)
onnx_program.save(os.path.join(base_dir, "SymbolCNN.onnx"))
onnx_program.save(os.path.join(base_dir, "application", "SymbolCNN.onnx"))
