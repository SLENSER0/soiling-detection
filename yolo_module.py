# yolo_module.py

import os
from ultralytics import YOLO
import numpy as np
import cv2

def load_yolo_model(model_path):
    """
    Загружает модель YOLO из указанного пути.
    
    :param model_path: Путь к файлу модели YOLO.
    :return: Загруженная модель YOLO.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель по пути {model_path} не найдена.")
    model = YOLO(model_path)
    return model

def predict_with_yolo(model, image_np, device=0):
    """
    Выполняет предсказание сегментации объектов на изображении с помощью модели YOLO.
    
    :param model: Загруженная модель YOLO.
    :param image_np: Изображение в формате NumPy массива.
    :param device: Устройство для выполнения (по умолчанию cuda 0).
    :return: Результаты предсказания.
    """
    results = model.predict(image_np, verbose=False, device=device)
    return results

def generate_random_color():
    """
    Генерирует случайный цвет в формате RGBA.

    :return: Кортеж с четырьмя значениями (R, G, B, A).
    """
    import random
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 100)

if __name__ == "__main__":
    from PIL import Image

    # Путь к весам YOLO
    model_path = "./weights/best433.pt"

    yolo_model = load_yolo_model(model_path)

    image_path = "./images/4_1709186522_0.png"
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    predictions = predict_with_yolo(yolo_model, img_np)

    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    results = predictions[0]
    if results.masks:
        masks = results.masks.data.cpu().numpy() 
        combined_mask = np.zeros((img_np.shape[0], img_np.shape[1], 4), dtype=np.uint8)  # RGBA

        for i, mask in enumerate(masks):
            color = generate_random_color()
            mask_resized = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_binary = (mask_resized > 0.5).astype(np.uint8)

            colored_mask = np.zeros_like(combined_mask)
            colored_mask[..., 0] = color[0]  # R
            colored_mask[..., 1] = color[1]  # G
            colored_mask[..., 2] = color[2]  # B
            colored_mask[..., 3] = color[3] * mask_binary  # A

            combined_mask = cv2.addWeighted(combined_mask, 1.0, colored_mask, 1.0, 0)

        combined_mask_image = Image.fromarray(combined_mask, mode="RGBA")
        mask_filename = os.path.splitext(os.path.basename(image_path))[0] + "_masks.png"
        mask_save_path = os.path.join(results_dir, mask_filename)
        combined_mask_image.save(mask_save_path)
        print(f"Объединенные маски сохранены по пути: {mask_save_path}")

    else:
        print("Нет обнаруженных масок.")
