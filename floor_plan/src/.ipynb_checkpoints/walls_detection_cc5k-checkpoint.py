import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from floortrans.models import get_model
from floortrans.post_prosessing import split_prediction, get_polygons
from floortrans.loaders import RotateNTurns
from floortrans.plotting import shp_mask

rot = RotateNTurns()

device = torch.device('cpu')
model = get_model('hg_furukawa_original', 51)

n_classes = 44
split = [21, 12, 11]
n_rooms = 12
model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
checkpoint = torch.load('models/weights/model_best_val_loss_var.pkl', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state'])

def predict_cubicasa5k(model, image_array, device):
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Ajuste o tamanho conforme necessário
        transforms.ToTensor(),  # Converte a imagem em tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normaliza conforme o modelo treinado
    ])
    
    image = Image.fromarray(image_array)
    height, width = image.size
    
    image = transform(image).unsqueeze(0)
    
    image = image.to(device)
    
    model.eval()
    
    with torch.no_grad():
        img_size = (height, width)
        
        rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
        pred_count = len(rotations)
        prediction = torch.zeros([pred_count, n_classes, height, width])
        for i, r in enumerate(rotations):
            forward, back = r
            # We rotate first the image
            rot_image = rot(image, 'tensor', forward)
            pred = model(rot_image)
            # We rotate prediction back
            pred = rot(pred, 'tensor', back)
            # We fix heatmaps
            pred = rot(pred, 'points', back)
            # We make sure the size is correct
            pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)
            # We add the prediction to output
            prediction[i] = pred[0]
    
    return prediction

def prediction_handler(image, predictions):
    
    prediction = torch.mean(predictions, 0, True)
    
    image_size = image.shape[:2]
    
    rooms_pred = F.softmax(prediction[0, 21:21+12], 0).cpu().data.numpy()
    rooms_pred = np.argmax(rooms_pred, axis=0)
    
    heatmaps, rooms, icons = split_prediction(prediction, image_size, split)
    polygons, types, room_polygons, room_types = get_polygons((heatmaps, rooms, icons), 0.1, [1, 2])
    
    mask = np.zeros(image_size, dtype=np.uint8)
    wall_type = 2 
    
    for i, pol in enumerate(room_polygons):
        pol_room_seg = np.zeros(image_size)
        tmp_mask = shp_mask(pol, np.arange(image_size[1]), np.arange(image_size[0]))
        pol_room_seg[tmp_mask] = 1
        image[pol_room_seg == 1] = (245, 245, 220)
    
    for polygon, polygon_type in zip(polygons, types):
        if polygon_type["class"] == wall_type:
            polygon = polygon.astype(np.int32)
            cv2.fillPoly(mask, [polygon], 1)
            cv2.fillPoly(image, [polygon], color=(0, 255, 0))

    return image

def get_walls(image):
    predictions = predict_cubicasa5k(model, image, "cpu")
    image_done = prediction_handler(image.copy(), predictions)

    return image_done