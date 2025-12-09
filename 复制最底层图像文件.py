import os
import shutil
from pathlib import Path


def copy_bottom_images_simple(source_folder, destination_folder):
    """
    简化版本：复制最底层的图像文件到新文件夹
    """
    # 确保目标文件夹存在
    os.makedirs(destination_folder, exist_ok=True)

    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}

    # 复制计数器
    copied_count = 0

    # 遍历所有子文件夹
    for root, dirs, files in os.walk(source_folder):
        # 如果当前文件夹没有子文件夹，说明是最底层
        if not dirs:
            # 复制所有图像文件
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    source_path = os.path.join(root, file)
                    destination_path = os.path.join(destination_folder, file)

                    # 处理文件名冲突
                    counter = 1
                    while os.path.exists(destination_path):
                        name, ext = os.path.splitext(file)
                        new_name = f"{name}_{counter}{ext}"
                        destination_path = os.path.join(destination_folder, new_name)
                        counter += 1

                    shutil.copy2(source_path, destination_path)
                    copied_count += 1
                    print(f"已复制: {file}")

    print(f"复制完成! 共复制 {copied_count} 个文件")
    return copied_count


# 使用示例
source = r"C:\Users\srt69\Desktop\111\896W-右前座椅"
destination = "ceshi"

copy_bottom_images_simple(source, destination)