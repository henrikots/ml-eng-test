import cv2
import numpy as np
import random
import easyocr

reader = easyocr.Reader(['en'])

def remove_small_components(mask, threshold_factor=0.5):
    # find all connected components (objects)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    new_mask = np.zeros(mask.shape, dtype=np.uint8)

    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0: 
        return mask

    mean_area = np.mean(areas)
    min_size = mean_area * threshold_factor

    # keep only the components that have area size > min_size
    for i in range(1, num_labels):  # Ignorar o fundo
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            new_mask[labels == i] = 255  # keep the component

    return new_mask


def remove_text_area(image, mask):
    
    c_mask = mask.copy()  # Cópia da máscara, se necessário

    results = reader.readtext(image)

    for coords, text, confidence in results:
        if confidence > 0.5:
            (top_left, top_right, bottom_right, bottom_left) = coords
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))

            cv2.rectangle(c_mask, top_left, bottom_right, color=0, thickness=-1)
            
    return c_mask

    
def create_mask(image, min_component_size=0.5, remove_text=False):

    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # first noise reduction
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    # binaryze the image with a theshold of 127
    _, binary_mask = cv2.threshold(blurred_image, 200, 255, cv2.THRESH_BINARY_INV)

    if remove_text:
        binary_mask = remove_text_area(image, binary_mask)

    # applying erosion and dilatation to remove noise as texts, non wall lines 
    kernel = np.ones((5, 5), np.uint8)
    binary_mask_clean = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask_clean = cv2.morphologyEx(binary_mask_clean, cv2.MORPH_CLOSE, kernel)
    # removing small components with a min area criteria
    binary_mask_clean = remove_small_components(binary_mask_clean, min_component_size)


    return binary_mask_clean


def get_straight_lines(mask):
    
    edges = cv2.Canny(mask, 10, 250, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=4, maxLineGap=50)
    lines_image = np.zeros_like(mask)

    img_h, img_w = mask.shape

    objs = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_image, (x1, y1), (x2, y2), 255, 2)  # Cor branca (255) com espessura 2

            x_min, y_min = min(x1, x2), min(y1, y2)
            x_max, y_max = max(x1, x2), max(y1, y2)

            width = (x_max - x_min) / img_w if (x_max - x_min) != 0 else 3 / img_w
            height = (y_max - y_min) / img_h if (y_max - y_min) != 0 else 3 / img_h
            x_center = (x_min + x_max) / 2 / img_w
            y_center = (y_min + y_max) / 2 / img_h

            x1_norm = x1 / img_w
            y1_norm = y1 / img_h
            x2_norm = x2 / img_w
            y2_norm = y2 / img_h

            objs.append(f"0 {x_center} {y_center} {width} {height} {x1_norm} {y1_norm} {x2_norm} {y2_norm}")

    final_mask = cv2.bitwise_and(mask, lines_image)

    return final_mask, objs

def draw_lines_from_labels(image, labels):
    
    img_h, img_w, _ = image.shape

    for label in labels:
        cls, x_center, y_center, width, height, x1, y1, x2, y2 = label.split(" ")

        x1_real = int(float(x1) * img_w)
        y1_real = int(float(y1) * img_h)
        x2_real = int(float(x2) * img_w)
        y2_real = int(float(y2) * img_h)

        cv2.line(image, (x1_real, y1_real), (x2_real, y2_real), (0, 255, 0), 3)
        cv2.circle(image, (x1_real, y1_real), 5, (255, 0, 0))
        cv2.circle(image, (x2_real, y2_real), 5, (255, 0, 0))

    return image


def get_walls(image, min_component_size=0.5, remove_text=False):

    if len(image.shape) == 3:  
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image.copy()
        
    mask = create_mask(image_gray, min_component_size, remove_text)
    final_mask, labels = get_straight_lines(mask)

    final_image = draw_lines_from_labels(image.copy(), labels)

    return final_image, labels



