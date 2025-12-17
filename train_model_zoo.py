import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import requests
import zipfile
import shutil
import random
import json
import argparse
from tqdm import tqdm

# --- 全局默认配置 ---
CONFIG = {
    "TINY_IMAGENET_URL": "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    "DATA_DIR": "tiny-imagenet-data",
    "ZIP_FILE": "tiny-imagenet-200.zip",
    "ROOT_DIR": "tiny-imagenet-200",
    "NUM_SUBSETS_TO_TRAIN": 200, 
    "NUM_CLASSES_SUBSET": 10,
    "BATCH_SIZE": 128,
    "NUM_EPOCHS": 10,
    "LEARNING_RATE": 2e-3,
    "NUM_WORKERS": 4,
    "OUTPUT_DIR": "model_zoo/resnet18_TinyImagenet",
    "SEED": 42,
    "VAL_SPLIT": 0.2,
    "DEVICE": "cpu"
}

# --- 1. 参数解析 ---
def parse_args():
    parser = argparse.ArgumentParser(description="Tiny-ImageNet Model Zoo Trainer")
    
    parser.add_argument("--num_subsets", type=int, default=200, help="要训练的子集总数")
    parser.add_argument("--classes_per_subset", type=int, default=10, help="每个子集的类别数量")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch Size")
    parser.add_argument("--lr", type=float, default=2e-3, help="学习率")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 线程数")
    parser.add_argument("--data_dir", type=str, default="tiny-imagenet-data", help="数据存储目录")
    parser.add_argument("--output_dir", type=str, default="model_zoo/resnet18_TinyImagenet", help="模型输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--val_split", type=float, default=0.2, help="验证集比例")
    # 支持 auto, cuda, cpu
    parser.add_argument("--device", type=str, default="auto", help="设备 (cuda, cpu, auto)")
    
    args = parser.parse_args()
    return args

def update_config(args):
    CONFIG["NUM_SUBSETS_TO_TRAIN"] = args.num_subsets
    CONFIG["NUM_CLASSES_SUBSET"] = args.classes_per_subset
    CONFIG["NUM_EPOCHS"] = args.epochs
    CONFIG["BATCH_SIZE"] = args.batch_size
    CONFIG["LEARNING_RATE"] = args.lr
    CONFIG["DATA_DIR"] = args.data_dir
    CONFIG["NUM_WORKERS"] = args.num_workers
    CONFIG["OUTPUT_DIR"] = args.output_dir
    CONFIG["SEED"] = args.seed
    CONFIG["VAL_SPLIT"] = args.val_split
    
    # [修改] 这里的 auto 逻辑：只检测 CUDA，否则 CPU
    if args.device == "auto":
        if torch.cuda.is_available():
            CONFIG["DEVICE"] = "cuda"
        else:
            CONFIG["DEVICE"] = "cpu"
    else:
        # 如果用户强制传 mps，这里也会接受，但 Shell 脚本不会传 mps 进来
        CONFIG["DEVICE"] = args.device

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- 2. 数据准备 ---
def download_and_unzip_tiny_imagenet():
    train_dir = os.path.join(CONFIG["DATA_DIR"], CONFIG["ROOT_DIR"], "train")
    if os.path.exists(train_dir): return train_dir

    os.makedirs(CONFIG["DATA_DIR"], exist_ok=True)
    zip_path = os.path.join(CONFIG["DATA_DIR"], CONFIG["ZIP_FILE"])

    if not os.path.exists(zip_path):
        print(f"正在下载 {CONFIG['TINY_IMAGENET_URL']}...")
        try:
            with requests.get(CONFIG["TINY_IMAGENET_URL"], stream=True) as r:
                r.raise_for_status()
                with open(zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        except Exception: return None

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref: zip_ref.extractall(CONFIG["DATA_DIR"])
    except Exception: return None
    return train_dir

def create_subset_dataset(all_classes, selected_classes, train_root):
    temp_dir = "temp_subset_dataset"
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    for class_name in selected_classes:
        src_dir = os.path.join(train_root, class_name)
        shutil.copytree(src_dir, os.path.join(temp_dir, class_name))
    return temp_dir

# --- 3. 训练逻辑 ---
def train_classifier_head(device, train_loader, val_loader, subset_key, output_dir):
    print(f"\n--- 正在训练 {subset_key} ---")
    
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters(): param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, CONFIG["NUM_CLASSES_SUBSET"])
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=CONFIG["LEARNING_RATE"])
    
    best_accuracy = -1.0  
    save_path = os.path.join(output_dir, f"resnet18_head_{subset_key}.pth")
    
    for epoch in range(CONFIG["NUM_EPOCHS"]):
        model.train() 
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        model.eval() 
        total_correct = 0
        total_samples = 0
        with torch.no_grad(): 
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1) 
                total_samples += labels.size(0)
                total_correct += (predicted == labels).sum().item()
        
        accuracy = 100.0 * total_correct / total_samples
        print(f"[Epoch {epoch+1}/{CONFIG['NUM_EPOCHS']}] Loss: {running_loss/len(train_loader):.3f} | Acc: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.fc.state_dict(), save_path)

    print(f"-> {subset_key} 训练完成。最佳准确率: {best_accuracy:.2f}%")
    return best_accuracy

# --- 4. 主函数 ---
def main():
    args = parse_args()
    update_config(args)
    set_seed(CONFIG["SEED"])

    device = torch.device(CONFIG["DEVICE"])
    print(f"正在使用设备: {device}")
    
    train_root = download_and_unzip_tiny_imagenet()
    if train_root is None: return
    
    all_classes = [d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))]
    
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    output_dir = CONFIG["OUTPUT_DIR"]
    os.makedirs(output_dir, exist_ok=True)
    mapping_file_path = os.path.join(output_dir, "subset_class_mapping.json")

    all_subset_mappings = {}
    if os.path.exists(mapping_file_path):
        try:
            with open(mapping_file_path, 'r') as f: all_subset_mappings = json.load(f)
        except: all_subset_mappings = {}

    for i in range(CONFIG["NUM_SUBSETS_TO_TRAIN"]):
        subset_index = i + 1
        subset_key = f"subset_{subset_index}"
        temp_subset_dir = None
        try:
            if subset_key in all_subset_mappings:
                if os.path.exists(os.path.join(output_dir, f"resnet18_head_{subset_key}.pth")):
                    print(f"跳过 {subset_key} (已完成)")
                    continue
                else:
                    selected_classes = all_subset_mappings[subset_key]["classes"]
            else:
                selected_classes = random.sample(all_classes, CONFIG["NUM_CLASSES_SUBSET"])
                all_subset_mappings[subset_key] = {"classes": selected_classes, "best_accuracy": -1.0}
                with open(mapping_file_path, 'w') as f: json.dump(all_subset_mappings, f, indent=4)

            temp_subset_dir = create_subset_dataset(all_classes, selected_classes, train_root)
            subset_dataset = datasets.ImageFolder(temp_subset_dir, transform=data_transforms)
            
            val_size = int(len(subset_dataset) * CONFIG["VAL_SPLIT"])
            train_dataset, val_dataset = torch.utils.data.random_split(
                subset_dataset, [len(subset_dataset) - val_size, val_size], 
                generator=torch.Generator().manual_seed(CONFIG["SEED"])
            )
            
            train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=CONFIG["NUM_WORKERS"])
            val_loader = DataLoader(val_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=CONFIG["NUM_WORKERS"])
            
            best_acc = train_classifier_head(device, train_loader, val_loader, subset_key, output_dir)
            
            if best_acc > 0:
                all_subset_mappings[subset_key]["best_accuracy"] = round(best_acc, 2)
                with open(mapping_file_path, 'w') as f: json.dump(all_subset_mappings, f, indent=4)
            
        except Exception as e:
            print(f"错误 ({subset_key}): {e}")
        finally:
            if temp_subset_dir and os.path.exists(temp_subset_dir): shutil.rmtree(temp_subset_dir)

if __name__ == "__main__":
    main()