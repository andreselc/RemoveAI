import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize
from PIL import Image
from transformers import AutoModelForImageSegmentation
import torch.nn.functional as F

# Cargar el modelo
model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Definir funciones de preprocesamiento y postprocesamiento
def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), size=model_input_size, mode='bilinear')
    image = torch.divide(im_tensor, 255.0)
    image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    return image

def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear'), 0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    im_array = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    im_array = np.squeeze(im_array)
    return im_array

# Función para procesar un video
def process_video(input_video_path, output_video_path, model, model_input_size):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))), True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        orig_im_size = frame.shape[0:2]
        image = preprocess_image(frame, model_input_size).to(device)

        # Inferencia
        result = model(image)

        # Post-proceso
        result_image = postprocess_image(result[0][0], orig_im_size)

        # Crear una imagen con fondo transparente
        pil_im = Image.fromarray(result_image)
        no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
        orig_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        no_bg_image.paste(orig_image, mask=pil_im)

        # Convertir de vuelta a formato de OpenCV
        no_bg_image = cv2.cvtColor(np.array(no_bg_image), cv2.COLOR_RGBA2BGRA)

        out.write(no_bg_image)

    cap.release()
    out.release()

# Definir el tamaño de entrada del modelo
model_input_size = [512, 512]  # Ajusta este tamaño según las necesidades de tu modelo

# Procesar el video
input_video_path = 'input_video.mp4'
output_video_path = 'output_video.avi'
process_video(input_video_path, output_video_path, model, model_input_size)