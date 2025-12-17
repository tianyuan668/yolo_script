import os
import json
import shutil
from pathlib import Path
import random

import numpy as np

def split_yolo_dataset(image_folder, label_folder, output_folder, train_ratio=0.8, copy_files=True):
    """
    将YOLO格式的数据集划分为训练集和验证集

    Parameters:
    - image_folder: 原始图片文件夹路径
    - label_folder: 原始标签文件夹路径（YOLO格式的txt文件）
    - output_folder: 输出文件夹路径
    - train_ratio: 训练集比例，默认0.8（80%训练，20%验证）
    - copy_files: True为复制文件，False为移动文件
    """
    # 创建输出目录结构
    os.makedirs(os.path.join(output_folder, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'labels', 'val'), exist_ok=True)

    # 获取所有图片文件名（支持常见图片格式）
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(image_folder)
                   if os.path.splitext(f)[1].lower() in image_extensions]

    # 随机打乱文件顺序
    random.shuffle(image_files)

    # 计算训练集和验证集的数量
    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    # 文件操作函数（复制或移动）
    file_operation = shutil.copy2 if copy_files else shutil.move

    # 处理训练集
    for image_file in train_files:
        image_name = os.path.splitext(image_file)[0]
        label_file = image_name + '.txt'

        # 源路径
        image_src = os.path.join(image_folder, image_file)
        label_src = os.path.join(label_folder, label_file)

        # 目标路径
        image_dst = os.path.join(output_folder, 'images', 'train', image_file)
        label_dst = os.path.join(output_folder, 'labels', 'train', label_file)

        # 处理图片文件
        if os.path.exists(image_src):
            file_operation(image_src, image_dst)

        # 处理对应的标签文件（如果存在）
        if os.path.exists(label_src):
            file_operation(label_src, label_dst)
        else:
            print(f"警告: 未找到标签文件 {label_file}")

    # 处理验证集
    for image_file in val_files:
        image_name = os.path.splitext(image_file)[0]
        label_file = image_name + '.txt'

        # 源路径
        image_src = os.path.join(image_folder, image_file)
        label_src = os.path.join(label_folder, label_file)

        # 目标路径
        image_dst = os.path.join(output_folder, 'images', 'val', image_file)
        label_dst = os.path.join(output_folder, 'labels', 'val', label_file)

        # 处理图片文件
        if os.path.exists(image_src):
            file_operation(image_src, image_dst)

        # 处理对应的标签文件（如果存在）
        if os.path.exists(label_src):
            file_operation(label_src, label_dst)
        else:
            print(f"警告: 未找到标签文件 {label_file}")

    print(f"数据集划分完成！")
    print(f"训练集: {len(train_files)} 张图片")
    print(f"验证集: {len(val_files)} 张图片")
    print(f"输出路径: {output_folder}")

class PolygonToYOLOConverter:
    def __init__(self, class_mapping):
        self.class_mapping = class_mapping
        self.leak_counter = 0  # 用于统计漏检样本

    def _get_bbox(self, points):
        """从多边形点集计算最小外接矩形"""
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        xmin = min(x_coords)
        ymin = min(y_coords)
        xmax = max(x_coords)
        ymax = max(y_coords)
        return xmin, ymin, xmax, ymax

    def _normalize_coordinates(self, xmin, ymin, xmax, ymax, img_width, img_height):
        """坐标归一化处理"""
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        return x_center, y_center, width, height

    def convert_single_json(self, json_path, output_dir, img_width, img_height):
        """转换单个JSON文件"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 验证必要字段
            if 'shapes' not in data or 'imageWidth' not in data or 'imageHeight' not in data:
                print(f"无效的JSON格式: {json_path}")
                return False

            # 创建输出文件路径
            img_name = Path(json_path).stem
            txt_path = Path(output_dir) /  f"{img_name}.txt"
            os.makedirs(txt_path.parent, exist_ok=True)

            with open(txt_path, 'w', encoding='utf-8') as f:
                for shape in data['shapes']:
                    label = shape.get('label', '')
                    if label not in self.class_mapping:
                        print(f"警告: 未定义的类别 '{label}' 在文件 {json_path} 中，已跳过")
                        continue

                    class_id = self.class_mapping[label]
                    points = np.array(shape['points'], dtype=np.float32)

                    # 计算边界框
                    xmin, ymin, xmax, ymax = self._get_bbox(points)
                    x_center, y_center, width, height = self._normalize_coordinates(
                        xmin, ymin, xmax, ymax, img_width, img_height
                    )

                    # 过滤无效框（避免除零错误和越界）
                    if width <= 0 or height <= 0 or not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                        print(f"警告: 无效边界框在 {json_path} 中，已跳过")
                        continue

                    # 写入YOLO格式
                    line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    f.write(line + '\n')

            return True
        except Exception as e:
            print(f"处理 {json_path} 时发生错误: {str(e)}")
            return False

    def batch_convert(self, json_dir, output_dir, img_width, img_height):
        """批量转换JSON文件夹"""
        print(f"开始批量转换: {json_dir} -> {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # # 创建标准YOLO目录结构
        # for subset in ['train', 'val', 'test']:
        #     subset_dir = Path(output_dir) / subset
        #     subset_dir.mkdir(exist_ok=True)
        #     (subset_dir / "images").mkdir(exist_ok=True)
        #     (subset_dir / "labels").mkdir(exist_ok=True)

        # 处理所有JSON文件
        json_files = [f for f in os.listdir(json_dir) if f.lower().endswith('.json')]
        total = len(json_files)
        success = 0
        failed = 0

        for i, json_file in enumerate(json_files, 1):
            json_path = os.path.join(json_dir, json_file)
            img_name = Path(json_path).stem
              # 自定义划分逻辑

            # 复制图像文件
            img_src = os.path.join(json_dir, f"{img_name}.jpg")
            if not os.path.exists(img_src):
                img_src = os.path.join(json_dir, f"{img_name}.png")
            if os.path.exists(img_src):
                img_dst = os.path.join(output_dir, f"{img_name}.jpg")
                os.makedirs(os.path.dirname(img_dst), exist_ok=True)
                shutil.copy(img_src, img_dst)
            else:
                print(f"警告: 未找到图像文件 {img_src}，跳过转换")
                continue

            # 转换标注
            if self.convert_single_json(json_path, output_dir, img_width, img_height):
                success += 1
                print(f"[{i}/{total}] 成功转换: {json_file} -> {img_name}.txt")
            else:
                failed += 1

        print(f"\n转换完成！成功: {success}/{total}，失败: {failed}")

    def _get_subset(self, filename):
        """根据文件名划分数据集（示例：按8:1:1划分）"""
        # 这里可以根据实际需求修改划分逻辑
        import random
        random.seed(42)
        rand_num = random.random()
        if rand_num < 0.8:
            return 'train'
        elif rand_num < 0.9:
            return 'val'
        else:
            return 'test'


if __name__ == "__main__":
    # 直接在代码中设置参数
    json_dir = r"E:\data\气孔"  #输入路径
    temp_dir = r"D:\Users\Win10\Desktop\暂存\牌照美纹纸-65张json" #临时路径
    output_folder = r"E:\data\气孔_1217"  # 替换为您想要的输出路径
    os.makedirs(temp_dir, exist_ok=True)
    img_width = 2448
    img_height = 2048

    # 定义类别映射（根据实际标注修改）
    class_mapping = {
        "0": 0,
        '1':1,
        "2":2,
    }

    converter = PolygonToYOLOConverter(class_mapping)
    converter.batch_convert(json_dir, temp_dir, img_width, img_height)


    # temp_dir = r"D:\Users\Win10\Desktop\youqianmen_mubiao\test"

    image_folder = temp_dir  # 替换为您的图片文件夹路径
    label_folder = temp_dir  # 替换为您的标签文件夹路径



    # 调用函数划分数据集
    split_yolo_dataset(
        image_folder=image_folder,
        label_folder=label_folder,
        output_folder=output_folder,
        train_ratio=0.8,  # 80%训练，20%验证
        copy_files=True  # True: 复制文件（推荐）；False: 移动文件
    )
    shutil.rmtree(temp_dir)



