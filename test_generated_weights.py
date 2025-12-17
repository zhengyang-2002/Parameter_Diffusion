import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

class SpecificClassDataset(Dataset):
    """
    一个专门的数据集类，确保标签(Labels)与用户输入的类别顺序完全一致。
    例如：classes=['n01', 'n02'] -> n01的图label=0, n02的图label=1
    """
    def __init__(self, tinyimagenet_dir, class_list, transform=None):
        self.tinyimagenet_dir = tinyimagenet_dir
        self.class_list = class_list
        self.transform = transform
        self.data = []

        # 尝试寻找 train 目录
        train_dir = os.path.join(tinyimagenet_dir, 'train')
        if not os.path.exists(train_dir):
            train_dir = tinyimagenet_dir # 兼容解压结构
            
        print(f"正在从 {train_dir} 加载 {len(class_list)} 个类别的数据...")

        for idx, class_name in enumerate(class_list):
            class_path = os.path.join(train_dir, class_name, 'images')
            if not os.path.exists(class_path):
                 # 尝试另一层结构
                class_path = os.path.join(train_dir, class_name)
            
            if not os.path.exists(class_path):
                print(f"警告: 找不到类别 {class_name} 的路径: {class_path}")
                continue

            # 获取所有图片
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
            for img_name in images:
                self.data.append((os.path.join(class_path, img_name), idx))
        
        print(f"共加载 {len(self.data)} 张图片。")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 224, 224), label

def main():
    parser = argparse.ArgumentParser(description="测试生成的权重")
    parser.add_argument("--weights_file", type=str, default="./generated_weights.pth", help="生成的权重文件路径")
    parser.add_argument("--tinyimagenet_dir", type=str, default="./tiny-imagenet-data/tiny-imagenet-200", help="数据集根目录")
    parser.add_argument("--classes", type=str, required=True, help="逗号分隔的类别列表 (必须与 Step 5 保持一致!)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda") # Mac用户可用 mps
    
    args = parser.parse_args()
    
    # 处理设备
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'mps' else "cpu")
    print(f"Using device: {device}")

    # 1. 解析类别
    class_list = [c.strip() for c in args.classes.split(',') if c.strip()]
    num_classes = len(class_list)
    print(f"测试类别 ({num_classes}类): {class_list}")

    # 2. 准备数据
    # ResNet18 标准输入是 224x224
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = SpecificClassDataset(args.tinyimagenet_dir, class_list, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 3. 加载生成的权重
    print(f"加载权重: {args.weights_file}")
    checkpoint = torch.load(args.weights_file, map_location=device)
    
    # 提取权重和偏置
    # sample_weights.py 保存格式是 {'weight': ..., 'bias': ...}
    if 'weight' in checkpoint and 'bias' in checkpoint:
        w = checkpoint['weight']
        b = checkpoint['bias']
    else:
        # 兼容只保存了 generated_weights 张量的情况
        print("未检测到标准格式，尝试从 raw tensor 解析...")
        gen_w = checkpoint['generated_weights'][0] # 取第一组
        # 假设前 num_classes * 512 是权重，后面是 bias
        w = gen_w[:num_classes*512].view(num_classes, 512)
        b = gen_w[num_classes*512:]
    
    print(f"权重形状: {w.shape}, 偏置形状: {b.shape}")

    # 4. 构建模型
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # 冻结骨干网络 (可选，这里只做推理，不反向传播，其实无所谓)
    for param in model.parameters():
        param.requires_grad = False
    
    # 替换分类头
    model.fc = nn.Linear(512, num_classes)
    model.fc.weight.data = w.to(device)
    model.fc.bias.data = b.to(device)
    
    model = model.to(device)
    model.eval()

    # 5. 开始评估
    correct = 0
    total = 0
    
    print("\n开始评估...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"\n========================================")
    print(f"测试结果:")
    print(f"总样本数: {total}")
    print(f"正确预测: {correct}")
    print(f"准确率 (Accuracy): {acc:.2f}%")
    print(f"========================================")

if __name__ == "__main__":
    main()