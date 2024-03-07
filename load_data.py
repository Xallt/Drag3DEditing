from huggingface_hub import snapshot_download
import os

if __name__ == '__main__':
    data_path = snapshot_download(repo_id="Yiwen-ntu/GaussianEditor_Result", repo_type="dataset")
    os.system(f"!rsync -a --copy-links {data_path}/* ./data")