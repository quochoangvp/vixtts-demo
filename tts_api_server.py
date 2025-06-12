from flask import Flask, request, jsonify, send_file
import torch
import torchaudio
import os
import uuid
import time
import string
import traceback
import logging
from queue import Queue
from threading import Thread
from datetime import datetime
from unidecode import unidecode
from underthesea import sent_tokenize
from huggingface_hub import snapshot_download
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from vinorm import TTSnorm

# Constants
MODEL_DIR = "/opt/vixtts-demo/model"
OUTPUT_DIR = "/opt/vixtts-demo/output"
REFERENCE_AUDIO = "assets/nu-luu-loat.wav"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("tts_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Cache
conditioning_latents_cache = {}

# Request queue
request_queue = Queue()
response_dict = {}

# Normalize text
def normalize_vietnamese_text(text):
    return (
        TTSnorm(text, unknown=False, lower=False, rule=True)
        .replace("..", ".")
        .replace("!.", "!")
        .replace("?.", "?")
        .replace(" .", ".")
        .replace(" ,", ",")
        .replace('"', "")
        .replace("'", "")
        .replace("AI", "Ây Ai")
        .replace("A.I", "Ây Ai")
    )

# Filename generator
def get_file_name(text, max_char=50):
    filename = text[:max_char].lower().replace(" ", "_")
    filename = filename.translate(str.maketrans("", "", string.punctuation.replace("_", "")))
    filename = unidecode(filename)
    current_datetime = datetime.now().strftime("%m%d%H%M%S")
    return f"{current_datetime}_{filename}.wav"

# Load model
def load_xtts_model():
    logger.info("⏳ Đang tải mô hình từ HuggingFace...")
    snapshot_download(repo_id="capleaf/viXTTS", repo_type="model", local_dir=MODEL_DIR)
    config_path = os.path.join(MODEL_DIR, "config.json")
    config = XttsConfig()
    config.load_json(config_path)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=MODEL_DIR)
    if torch.cuda.is_available():
        model.cuda()
        logger.info("✅ Model đã sử dụng CUDA.")
    else:
        logger.warning("⚠️ Model đang chạy bằng CPU.")
    return model

logger.info("🚀 Loading model...")
XTTS_MODEL = load_xtts_model()
logger.info("✅ Model loaded.")

# TTS Function
def run_tts(text, lang="vi", speaker_audio=REFERENCE_AUDIO, normalize=True):
    logger.info("🧠 Bắt đầu synthesize toàn bộ văn bản...")
    if normalize:
        text = normalize_vietnamese_text(text)
        logger.info(f"✅ Văn bản sau normalize: {text[:80]}...")

    cache_key = (speaker_audio, XTTS_MODEL.config.gpt_cond_len, XTTS_MODEL.config.max_ref_len)
    if cache_key in conditioning_latents_cache:
        gpt_latent, speaker_embed = conditioning_latents_cache[cache_key]
        logger.info("✅ Dùng conditioning từ cache.")
    else:
        gpt_latent, speaker_embed = XTTS_MODEL.get_conditioning_latents(
            audio_path=speaker_audio,
            gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
            max_ref_length=XTTS_MODEL.config.max_ref_len,
            sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
        )
        conditioning_latents_cache[cache_key] = (gpt_latent, speaker_embed)
        logger.info("✅ Conditioning latents mới tạo.")

    wav = XTTS_MODEL.inference(
        text=text,
        language=lang,
        gpt_cond_latent=gpt_latent,
        speaker_embedding=speaker_embed,
        temperature=0.3,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=30,
        top_p=0.85,
        enable_text_splitting=True,
    )["wav"]

    return torch.tensor(wav).unsqueeze(0)

# Worker thread

def worker():
    while True:
        request_id, data = request_queue.get()
        try:
            logger.info(f"🔧 Đang xử lý request: {request_id}")
            audio_tensor = run_tts(
                data["text"],
                data.get("lang", "vi"),
                data.get("speaker_audio", REFERENCE_AUDIO),
                data.get("normalize", True),
            )
            file_name = data.get("file_name") or get_file_name(data["text"])
            out_path = os.path.join(OUTPUT_DIR, file_name)
            torchaudio.save(out_path, audio_tensor, 24000)
            response_dict[request_id] = {"status": "done", "file": out_path}
            logger.info(f"✅ Audio đã lưu: {out_path}")
        except Exception as e:
            logger.error(f"❌ Lỗi khi xử lý request {request_id}: {e}", exc_info=True)
            response_dict[request_id] = {"status": "error", "error": str(e)}
        finally:
            request_queue.task_done()

# Start worker
Thread(target=worker, daemon=True).start()

# Flask app
app = Flask(__name__)

@app.route("/tts", methods=["POST"])
def tts_api():
    try:
        data = request.json
        text = data.get("text")
        if not text:
            return jsonify({"error": "Missing 'text' field."}), 400

        request_id = str(uuid.uuid4())
        logger.info(f"📥 Nhận request {request_id}, đưa vào hàng đợi...")
        request_queue.put((request_id, data))

        while request_id not in response_dict:
            time.sleep(0.5)

        result = response_dict.pop(request_id)

        if result["status"] == "done":
            return send_file(result["file"], mimetype="audio/wav")
        else:
            return jsonify({"error": result["error"]}), 500

    except Exception as e:
        logger.error("❌ Lỗi không xác định khi xử lý API", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Run the API server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)

