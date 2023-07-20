from facebook_resnet50 import facbook_RN_50
import torch
import os

model = facbook_RN_50()
weights_path = './weights_folder/epoch_1.pt'
model.load_state_dict(torch.load(weights_path))
model = torch.nn.Sequential(*list(model.children())[:-2])

save_path = 'final_model/model_evaluation/image_model.pt'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)

