import torch
from huggingface_hub import hf_hub_download

from moshi.models import LMGen, loaders


def run_local_moshi():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #   Download/Load the Moshi model
    moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
    moshi = loaders.get_moshi_lm(moshi_weight, device=device)

    # Load the Mimi audio codec (handles audio compression)
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=device)

    # Create a generator
    lm_gen = LMGen(moshi, temp=0.8, temp_text=0.7)

    print("Moshi loaded! You can now stream audio into `mimi` and `lm_gen`.")


if __name__ == "__main__":
    run_local_moshi()
