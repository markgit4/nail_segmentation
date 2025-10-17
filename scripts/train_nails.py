# train_nails.py
import os
from pathlib import Path
from ultralytics import YOLO

# --------------------------
# 配置数据集路径
# --------------------------
BASE_DIR = Path("E:/ultralytics-main/archive")
IMAGES_DIR_TRAIN = BASE_DIR / "images/train"
IMAGES_DIR_VAL   = BASE_DIR / "images/val"
LABELS_DIR_TRAIN = BASE_DIR / "labels/train"
LABELS_DIR_VAL   = BASE_DIR / "labels/val"

# --------------------------
# 生成 YOLO 数据集 yaml
# --------------------------
DATA_YAML = Path("E:/ultralytics-main/nails.yaml")
if not DATA_YAML.exists():
    data_yaml_content = f"""
train: {IMAGES_DIR_TRAIN.as_posix()}
val:   {IMAGES_DIR_VAL.as_posix()}

nc: 1
names: ['nail']
"""
    DATA_YAML.write_text(data_yaml_content)
    print(f"✅ nails.yaml 已生成: {DATA_YAML}")

# --------------------------
# 检查数据集完整性
# --------------------------
def check_dataset(images_dir, labels_dir):
    images = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    labels = sorted(labels_dir.glob("*.txt"))
    if len(images) != len(labels):
        print(f"⚠️ 图片与标签数量不一致: {len(images)} vs {len(labels)}")
    else:
        print(f"✅ {len(images)} 张图片都有对应标签。")
    return images, labels

print("==== 检查训练集 ====")
train_images, train_labels = check_dataset(IMAGES_DIR_TRAIN, LABELS_DIR_TRAIN)
print("==== 检查验证集 ====")
val_images, val_labels = check_dataset(IMAGES_DIR_VAL, LABELS_DIR_VAL)

# --------------------------
# 训练 YOLOv11-seg
# --------------------------
def train_yolo():
    # 使用官方预训练 YOLOv11n-seg 模型
    model = YOLO("yolo11n-seg.pt")

    model.train(
        data="E:/ultralytics-main/nails.yaml",
        epochs=50,
        batch=1,
        imgsz=(512, 640),  # 随机选择尺寸
        multi_scale=True,  # 启用随机尺寸
        rect=True,  # 保持长宽比
        workers=0,
        project="runs/segment/nail_seg_exp",
        name="nail_seg_multiscale",
        pretrained=True
    )

# --------------------------
# Windows 安全启动
# --------------------------
if __name__ == "__main__":
    train_yolo()
