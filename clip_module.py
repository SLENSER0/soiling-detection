import os
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

def load_clip_model(model_name="zer0int/CLIP-GmP-ViT-L-14"):
    """
    Загружает модель CLIP и процессор из указанного имени модели.
    
    :param model_name: Имя модели CLIP.
    :return: Загруженная модель, процессор и устройство.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    return clip_model, clip_processor, device

def load_class_embeddings(csv_path):
    """
    Загружает и обрабатывает CSV-файл с эмбеддингами классов.
    
    :param csv_path: Путь к CSV-файлу.
    :return: DataFrame с эмбеддингами и список названий колонок эмбеддингов.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Файл {csv_path} не найден.")
    df = pd.read_csv(csv_path)
    label_columns = [col for col in df.columns if col not in ['image_name', 'label']]
    class_embeddings = df.groupby('label')[label_columns].mean().reset_index()
    return class_embeddings, label_columns

def get_image_embedding(image, model, processor, device):
    """
    Получает эмбеддинг изображения с помощью модели CLIP.
    
    :param image: PIL изображение.
    :param model: Загруженная модель CLIP.
    :param processor: Процессор CLIP.
    :param device: Устройство для вычислений.
    :return: Нормализованный эмбеддинг изображения.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_emb = model.get_image_features(**inputs)
    image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)
    return image_emb.cpu().numpy()

def determine_class(image_emb, class_embeds, class_labels):
    """
    Определяет класс загрязнения на основе сходства эмбеддингов.
    
    :param image_emb: Эмбеддинг изображения.
    :param class_embeds: Эмбеддинги классов.
    :param class_labels: Список меток классов.
    :return: Название класса и значение сходства.
    """
    similarities = cosine_similarity(image_emb, class_embeds)
    most_similar_idx = np.argmax(similarities)
    return class_labels[most_similar_idx], similarities[0][most_similar_idx]

def determine_label(filename):
    """
    Определяет метку класса на основе имени файла.
    
    :param filename: Имя файла изображения.
    :return: Название класса.
    """
    base_name = os.path.splitext(filename)[0].lower()
    if base_name and base_name[0].isdigit():
        return "waterdrop"
    elif base_name.startswith("gr"):
        return "dirty"
    elif base_name.startswith("f"):
        return "F_label"  
    elif base_name.startswith("nj"):
        return "NJ_label"  
    elif base_name.startswith("gskm"):
        return "GSKM_label" 
    else:
        return "unknown"

if __name__ == "__main__":
    # Загрузка модели CLIP
    clip_model, clip_processor, device = load_clip_model()

    # Загрузка эмбеддингов классов
    embeddings_path = "embeddings_with_labels.csv"
    class_embeddings_df, embedding_columns = load_class_embeddings(embeddings_path)
    class_labels = class_embeddings_df['label'].tolist()
    class_embeds = class_embeddings_df[embedding_columns].values

    # Загрузка изображения
    image_path = "/home/ubuntu/soiling_detection/images/4_1709186522_0.png"
    img = Image.open(image_path).convert("RGB")

    # Получение эмбеддинга изображения
    image_emb = get_image_embedding(img, clip_model, clip_processor, device)

    label, similarity = determine_class(image_emb, class_embeds, class_labels)
    print(f"Определённый класс: {label} с похожестью {similarity:.2f}")
