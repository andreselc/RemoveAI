import os
import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize
from PIL import Image
from transformers import AutoModelForImageSegmentation
import torch.nn.functional as F

# Cargar el modelo
print("Cargando el modelo...")
model = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4", trust_remote_code=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Modelo cargado y enviado a:", device)

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
    print(f"Abriendo el video de entrada: {input_video_path}")
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error al abrir el video de entrada: {input_video_path}")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), True)
    
    if not out.isOpened():
        print(f"Error al crear el video de salida: {output_video_path}")
        return

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Procesando cuadro {frame_count}")
        orig_im_size = frame.shape[0:2]
        image = preprocess_image(frame, model_input_size).to(device)

        # Inferencia
        with torch.no_grad():
            result = model(image)
        
        # Ajustar el acceso al resultado según la estructura real
        if isinstance(result, tuple):
            result = result[0]

        # Post-proceso
        result_image = postprocess_image(result[0], orig_im_size)

        # Crear una imagen con fondo transparente
        pil_im = Image.fromarray(result_image)
        no_bg_image = Image.new("RGBA", pil_im.size, (0, 0, 0, 0))
        orig_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        no_bg_image.paste(orig_image, mask=pil_im)

        # Convertir de vuelta a formato de OpenCV
        no_bg_image = cv2.cvtColor(np.array(no_bg_image), cv2.COLOR_RGBA2BGRA)

        # Verificar las dimensiones del cuadro
        if no_bg_image.shape[1] != frame_width or no_bg_image.shape[0] != frame_height:
            print(f"Error: Las dimensiones del cuadro {frame_count} no coinciden con el video de salida")
            print(f"Dimensiones del cuadro: {no_bg_image.shape[1]}x{no_bg_image.shape[0]}, Dimensiones esperadas: {frame_width}x{frame_height}")
            return
        
        # Convertir el frame a un formato compatible con VideoWriter (BGR)
        bgr_frame = cv2.cvtColor(no_bg_image, cv2.COLOR_BGRA2BGR)

        # Escribir el cuadro en el archivo de video
        out.write(bgr_frame)

    cap.release()
    out.release()
    print("Procesamiento de video completado.")
    
    # Crear archivo de marcador para indicar que el procesamiento ha terminado
    flag_file_path = output_video_path + ".done"
    with open(flag_file_path, 'w') as flag_file:
        flag_file.write('done')
