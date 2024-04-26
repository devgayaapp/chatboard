from huggingface_hub import hf_hub_url, cached_download
import joblib
from config import STABLE_DIFFUSION_MODEL_DIR

# model_id = "stabilityai/stable-diffusion-2"
# model_id = "stabilityai/stable-diffusion-2-1"



REPO_ID = "stabilityai/stable-diffusion-2"
FILENAME = STABLE_DIFFUSION_MODEL_DIR / "stable_diffusion_2.joblib"

model = joblib.load(cached_download(
    hf_hub_url(REPO_ID, str(FILENAME))
))