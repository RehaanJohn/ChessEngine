
from pathlib import Path
from huggingface_hub import snapshot_download

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "malecns"

REPO_ID = "fernandofernandes/fly-connectome-malecns-166k"


def download_connectome():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading MaleCNS v1.0 processed connectome...")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(DATA_DIR),
        allow_patterns=[
            "manifest.json",
            "graph/*",
            "neurons/*",
            "populations/*",
        ],
    )

    print(f"Dataset available at: {DATA_DIR}")
    return DATA_DIR


if __name__ == "__main__":
    download_connectome()
