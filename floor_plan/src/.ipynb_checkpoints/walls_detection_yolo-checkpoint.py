import cv2
from ultralytics import YOLO

model = YOLO("models/weights/yolo_keypoints_walls.pt")

def get_walls(image):
    
    results = model.predict(image, conf=0.0)[0]

    img_h, img_w, _ = image.shape
    
    print(image.shape)
    
    for r in results:
        bound_box = r.boxes.xyxy
        conf = r.boxes.conf.tolist() 
        keypoints = r.keypoints.xyn.tolist()[0]
        
    
        x1 = int(keypoints[0][0] * img_w)
        y1 = int(keypoints[0][1] * img_h)
        x2 = int(keypoints[1][0] * img_w)
        y2 = int(keypoints[1][1] * img_h)
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return image