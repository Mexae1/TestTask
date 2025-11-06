FROM python:3.12-slim

# Отключаем интерактивные подсказки apt
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

# Устанавливаем минимальные системные библиотеки, нужные для OpenCV/ffmpeg и сборки пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    wget \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем файл зависимостей (ожидаем requirements.txt в корне проекта)
COPY requirements.txt /app/requirements.txt

# Обновляем pip и устанавливаем PyTorch (CPU) из официального зеркала PyTorch (более стабильно в регионах с ограничениями)
RUN pip install --upgrade pip setuptools wheel
# Устанавливаем CPU-версию torch/torchvision/torchaudio (если нужен GPU — этот шаг изменить)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Устанавливаем остальные Python-зависимости
RUN pip install --no-cache-dir -r /app/requirements.txt --retries 10 --timeout 120

# Копируем весь проект в контейнер
COPY . /app

# Создаём каталоги для входных/выходных данных
RUN mkdir -p /app/input /app/output

# Отключаем GUI (предотвратить ошибки Qt/OpenCV внутри контейнера)
ENV QT_QPA_PLATFORM=offscreen
ENV DISPLAY=""

# Рекомендуется запускать скрипт, расположенный в app/main.py
# Если у тебя main.py лежит в корне — изменить команду на ["python","main.py"]
CMD ["python", "app/main.py"]
