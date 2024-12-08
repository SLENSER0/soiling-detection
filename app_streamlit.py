# app_streamlit.py

import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2
import os

from yolo_module import load_yolo_model, predict_with_yolo
from clip_module import load_clip_model, load_class_embeddings, get_image_embedding, determine_class

import pandas as pd

# Настройка заголовка приложения
st.title("Сегментация и Классификация загрязнений на Изображениях")

# Путь к моделям и эмбеддингам
yolo_model_path = "./weights/best433.pt" 
clip_embeddings_path = "embeddings_with_labels.csv"

# Загрузка моделей
@st.cache_resource
def initialize_models():
    yolo = load_yolo_model(yolo_model_path)
    clip_model, clip_processor, device = load_clip_model()
    class_embeddings_df, embedding_columns = load_class_embeddings(clip_embeddings_path)
    class_labels = class_embeddings_df['label'].tolist()
    class_embeds = class_embeddings_df[embedding_columns].values
    return yolo, clip_model, clip_processor, device, class_labels, class_embeds

yolo_model, clip_model, clip_processor, device, class_labels, class_embeds = initialize_models()

# Цветовая карта для классов
color_map = {
    "waterdrop": (0, 0, 255, 100),  
    "dirty": (0, 255, 0, 100),      
    "dust": (255, 255, 0, 100), 
    "unknown": (128, 128, 128, 100) 
}

# Загрузка изображения пользователем
st.write("Загрузите изображение, и модель выполнит сегментацию объектов и классификацию сегментов.")
uploaded_file = st.file_uploader("Выберите изображение для сегментации", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    height, width = img_np.shape[:2]

    st.subheader("Исходное изображение")
    st.image(img, caption="Загруженное изображение", use_container_width=True)

    # Выполнение сегментации с помощью YOLO
    with st.spinner('Выполняется сегментация...'):
        results = predict_with_yolo(yolo_model, img_np)

    segments_info = []  

    segmented_pil = img.convert("RGBA")
    overlay = Image.new("RGBA", segmented_pil.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    for result in results:
        if result.masks:
            for mask_i in result.masks:
                mask_data = mask_i.data.cpu().numpy()[0] 
                mask_resized = cv2.resize(mask_data, (width, height), interpolation=cv2.INTER_NEAREST)
                mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if len(contour) < 3:
                        continue

                    polygon = contour.squeeze().tolist()
                    if isinstance(polygon[0], int):
                        continue

                    mask_segment = Image.fromarray(mask_binary).convert("L")
                    cropped_img = Image.composite(img, Image.new("RGB", img.size, (0, 0, 0)), mask_segment)
                    
                    preprocessed = cropped_img.resize((224, 224), Image.LANCZOS)

                    # Получение эмбеддинга и определение класса
                    embed = get_image_embedding(preprocessed, clip_model, clip_processor, device)
                    label, similarity = determine_class(embed, class_embeds, class_labels)

                    segments_info.append({
                        "label": label,
                        "similarity": similarity
                    })

                    color = color_map.get(label, color_map["unknown"])

                    polygon_flat = [tuple(point) for point in polygon]
                    
                    draw.polygon(polygon_flat, fill=color)

    final_image = Image.alpha_composite(segmented_pil, overlay)

    st.subheader("Сегментированное изображение")
    st.image(final_image, caption="Сегментированное изображение", use_container_width=True)

    st.subheader("Информация о классах загрязнений")
    for idx, segment in enumerate(segments_info, start=1):
        st.write(f"**Загрязнение {idx}:**")
        st.write(f" - **Класс:** {segment['label']}")
        st.write(f" - **Сходство:** {segment['similarity']:.2f}")
        st.write("---")
