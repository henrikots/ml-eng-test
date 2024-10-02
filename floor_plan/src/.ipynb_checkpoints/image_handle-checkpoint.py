from pdf2image import convert_from_path
import numpy as np

def pdf_to_image_list(pdf_path, dpi=200):
    images = convert_from_path(pdf_path, dpi=dpi)
    
    return [np.array(image) for image in images]

def split_image_if_large(image, max_size=1000):

    height, width, _ = image.shape
    images = []

    if width <= max_size and height <= max_size:
        return [(image, (0, 0))]

    # Determina quantas divisões são necessárias para cada dimensão
    cols = (width + max_size - 1) // max_size  # Número de colunas
    rows = (height + max_size - 1) // max_size  # Número de linhas

    for row in range(rows):
        for col in range(cols):
            # Definir as coordenadas para o corte da imagem
            left = col * max_size
            upper = row * max_size
            right = min((col + 1) * max_size, width)
            lower = min((row + 1) * max_size, height)

            # Cortar a imagem
            cropped_image = image[upper:lower, left:right]
            images.append((cropped_image, (col, row)))

    return images

def join_images(images, original_width, original_height, max_size=1000):

    # Criar uma imagem em branco do tamanho original (3 canais para RGB)
    combined_image = np.zeros((original_height, original_width, 3), dtype=np.uint8)

    # Colar cada parte de volta na imagem combinada
    for img, (col, row) in images:
        left = col * max_size
        upper = row * max_size
        combined_image[upper:upper+img.shape[0], left:left+img.shape[1]] = img

    return combined_image