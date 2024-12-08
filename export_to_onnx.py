from model import SymbolCNN
import torch
import os
from symbols import symbols

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "augmented_models_3", "checkpoint_34.pth")
image_size = (32, 32)

model = SymbolCNN(num_classes=len(symbols))  # Initialize the model
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()  # Set the model to evaluation mode


model.eval()
x = torch.randn(1, 1, model.image_size[0], model.image_size[1])
width = torch.tensor([10])
height = torch.tensor([20])

onnx_program = torch.onnx.dynamo_export(model, x, width, height)
onnx_program.save(os.path.join(base_dir, "SymbolCNN.onnx"))
