from flask import Flask, request, jsonify, send_file
import torch
import torchaudio
import os
import uuid
from underthesea import sent_tokenize
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from huggingface_hub import snapshot_download
from vinorm import TTSnorm
from unidecode import unidecode
from datetime import datetime
import string

# Constants
MODEL_DIR = "/opt/vixtts-demo/model"
OUTPUT_DIR = "/opt/vixtts-demo/output"
REFERENCE_AUDIO = "assets/nu-luu-loat.wav"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Caching
conditioning_latents_cache = {}

def normalize_vietnamese_text(text):
    text = (
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
    return text

def get_file_name(text, max_char=50):
    filename = text[:max_char].lower().replace(" ", "_")
    filename = filename.translate(str.maketrans("", "", string.punctuation.replace("_", "")))
    filename = unidecode(filename)
    current_datetime = datetime.now().strftime("%m%d%H%M%S")
    return f"{current_datetime}_{filename}.wav"

def load_xtts_model():
    snapshot_download(repo_id="capleaf/viXTTS", repo_type="model", local_dir=MODEL_DIR)
    config_path = os.path.join(MODEL_DIR, "config.json")
    config = XttsConfig()
    config.load_json(config_path)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=MODEL_DIR)
    if torch.cuda.is_available():
        model.cuda()
    return model

print("Loading model...")
XTTS_MODEL = load_xtts_model()
print("Model loaded.")

def run_tts(text, lang="vi", speaker_audio=REFERENCE_AUDIO, normalize=True):
    if normalize:
        text = normalize_vietnamese_text(text)
    sentences = sent_tokenize(text)

    cache_key = (speaker_audio, XTTS_MODEL.config.gpt_cond_len, XTTS_MODEL.config.max_ref_len)
    if cache_key in conditioning_latents_cache:
        gpt_latent, speaker_embed = conditioning_latents_cache[cache_key]
    else:
        gpt_latent, speaker_embed = XTTS_MODEL.get_conditioning_latents(
            audio_path=speaker_audio,
            gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
            max_ref_length=XTTS_MODEL.config.max_ref_len,
            sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
        )
        conditioning_latents_cache[cache_key] = (gpt_latent, speaker_embed)

    chunks = []
    for sent in sentences:
        if sent.strip() == "": continue
        wav = XTTS_MODEL.inference(
            text=sent,
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
        chunks.append(torch.tensor(wav))

    audio = torch.cat(chunks).unsqueeze(0)
    return audio

# Flask app init
app = Flask(__name__)

@app.route("/tts", methods=["POST"])
def tts_api():
    try:
        data = request.json

        text = data.get("text")
        lang = data.get("lang", "vi")
        speaker_audio = data.get("speaker_audio", REFERENCE_AUDIO)
        normalize = data.get("normalize", True)
        file_name = data.get("file_name")

        if not text:
            return jsonify({"error": "Missing 'text' field."}), 400

        audio_tensor = run_tts(text, lang, speaker_audio, normalize)

        if not file_name:
            file_name = get_file_name(text)

        out_path = os.path.join(OUTPUT_DIR, file_name)
        torchaudio.save(out_path, audio_tensor, 24000)

        return send_file(out_path, mimetype="audio/wav")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the API server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
