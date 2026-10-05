FROM python:3.13-slim

# Hugging Face runs Docker Spaces as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# CPU-only torch: the CUDA build is ~10x bigger and the Space has no GPU
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# CLAP's text model (public) baked into the image, so a cold start doesn't download it
RUN python -c "from transformers import AutoTokenizer, ClapTextModelWithProjection as M; \
M.from_pretrained('laion/larger_clap_general'); AutoTokenizer.from_pretrained('laion/larger_clap_general')"

COPY --chown=user . .

ENV FMA_DATA_DIR=/home/user/app/data \
    API_PUBLIC_BASE="" \
    FRONTEND_DIST=/home/user/app/frontend/dist \
    HF_DATA_REPO=mrln77/tastebridge-audio \
    HF_AUDIO_REPO=mrln77/tastebridge-audio \
    HF_AUDIO_PREFIX=""

EXPOSE 7860
CMD ["python", "start.py"]
