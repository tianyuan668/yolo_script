import os
import random
import shutil


def copy_random_images(source_dir, target_dir, num_images=100):
    """
    从源文件夹中随机复制指定数量的图片到目标文件夹

    参数:
        source_dir (str): 源图像文件夹路径
        target_dir (str): 目标文件夹路径
        num_images (int): 要复制的图片数量，默认为100
    """
    # 检查源文件夹是否存在
    if not os.path.exists(source_dir):
        print(f"错误：源文件夹 '{source_dir}' 不存在")
        return False

    # 创建目标文件夹（如果不存在）
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"已创建目标文件夹: {target_dir}")

    # 获取源文件夹中所有图片文件[1,4](@ref)
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
    all_files = os.listdir(source_dir)
    image_files = [f for f in all_files if f.lower().endswith(image_extensions)]

    print(f"在源文件夹中找到 {len(image_files)} 张图片")

    # 检查图片数量是否足够
    if len(image_files) < num_images:
        print(f"警告：源文件夹中只有 {len(image_files)} 张图片，少于要求的 {num_images} 张")
        num_images = len(image_files)

    # 随机选择图片[1,6](@ref)
    selected_images = random.sample(image_files, num_images)

    # 复制选中的图片到目标文件夹[1,4](@ref)
    copied_count = 0
    for image_name in selected_images:
        source_path = os.path.join(source_dir, image_name)
        target_path = os.path.join(target_dir, image_name)

        shutil.copy2(source_path, target_path)  # copy2会保留文件元数据
        copied_count += 1

    print(f"成功复制 {copied_count} 张图片到 {target_dir}")
    print("随机选择的图片列表:")
    for i, img in enumerate(selected_images, 1):
        print(f"{i}. {img}")

    return True


if __name__ == "__main__":
    # 在这里设置您的路径
    source_directory = r"E:\data\低分数采图\低分数采图\左后减震器分类\减震器2白\05.图像存储"  # 替换为您的源图像文件夹路径
    target_directory = r"E:\data\低分数采图\低分数采图\左后减震器分类\减震器2白"  # 替换为您的目标文件夹路径
    number_to_copy = 150  # 要复制的图片数量

    # 执行复制操作
    copy_random_images(source_directory, target_directory, number_to_copy)