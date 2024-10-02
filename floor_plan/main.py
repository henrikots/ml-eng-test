from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
import io

from src import walls_detection_cc5k as wdc
from src import walls_detection_traditional as wdt
from src import walls_detection_yolo as wdy

app = FastAPI()

def save_image(file: UploadFile):
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file_location

def enforce_three_color_channels(image):
    if len(image.shape) > 2:
        image = image[:, :, :3]
        return image
    else:
        return np.stack((image, ) * 3, axis=-1)

def handle_image_data(image_data):
    
    pil_image = Image.open(io.BytesIO(image_data))
    image_array = np.array(pil_image)
    image_array = enforce_three_color_channels(image_array)

    return image_array
    
@app.post("/inference-run-classic")
async def get_walls_classic(image: UploadFile = File(...)):
    image_array = handle_image_data(await image.read())

    image_segmented, _ = wdt.get_walls(image_array)
    segmented_pil_image = Image.fromarray(image_segmented)

    buffer = io.BytesIO()
    segmented_pil_image.save(buffer, format="JPEG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/jpeg")

@app.post("/inference-run-classic-ocr")
async def get_walls_classic_ocr(image: UploadFile = File(...)):
    image_array = handle_image_data(await image.read())

    image_segmented, _ = wdt.get_walls(image_array, remove_text=True)
    segmented_pil_image = Image.fromarray(image_segmented)

    buffer = io.BytesIO()
    segmented_pil_image.save(buffer, format="JPEG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/jpeg")

@app.post("/inference-run-cc5k")
async def get_walls_cc5k(image: UploadFile = File(...)):
    image_array = handle_image_data(await image.read())

    image_segmented = wdc.get_walls(image_array)
    segmented_pil_image = Image.fromarray(image_segmented)

    buffer = io.BytesIO()
    segmented_pil_image.save(buffer, format="JPEG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/jpeg")

@app.post("/inference-run-yolo")
async def get_walls_cc5k(image: UploadFile = File(...)):
    image_array = handle_image_data(await image.read())

    image_segmented = wdy.get_walls(image_array)
    segmented_pil_image = Image.fromarray(image_segmented)

    buffer = io.BytesIO()
    segmented_pil_image.save(buffer, format="JPEG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/jpeg")
