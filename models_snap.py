from huggingface_hub import snapshot_download
import yaml
import os

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

os.environ["HF_HOME"] = "/home/mila/d/dane.malenfant/scratch/huggingface"

log_path = f"/home/mila/d/dane.malenfant/scratch/qwen_interactive_log.txt"
os.makedirs(os.path.dirname(log_path), exist_ok=True)

TEACHER_MODEL = config["teacher_model"]
STUDENT_MODEL = config["student_model"]

#snapshot_download(
  ##  repo_id=config["teacher_model"],
  #  local_dir="/home/mila/d/dane.malenfant/scratch/qwen30b_instruct",
  #  local_dir_use_symlinks=False
#)

print("Downloading student snapshot")

snapshot_download(
    repo_id=config["student_model"],
    local_dir="/home/mila/d/dane.malenfant/scratch/qwen3.5_9B",
    local_dir_use_symlinks=False
)