import os
import shutil



def copy_paired_files(source_folder, target_folder, image_exts=('.jpg', '.jpeg', '.png'), annotation_ext='.json'):
    """
    将源文件夹中同名的图像文件和JSON标注文件一起复制到目标文件夹。

    Args:
        source_folder (str): 源文件夹路径，包含混合的图像和JSON文件。
        target_folder (str): 目标文件夹路径，用于存放配对成功的文件。
        image_exts (tuple): 图像文件的扩展名元组，例如 ('.jpg', '.png')。
        annotation_ext (str): 标注文件的扩展名，例如 '.json'。
    """

    # 创建目标文件夹（如果不存在）
    os.makedirs(target_folder, exist_ok=True)

    # 获取源文件夹下所有文件列表
    all_files = os.listdir(source_folder)

    # 分离出图像文件和JSON文件的基本名（不含扩展名）
    image_basenames = set()  # 存储图像文件的基本名
    json_basenames = set()  # 存储JSON文件的基本名

    # 用于记录文件名和完整路径的映射，便于后续复制
    file_path_map = {}

    for file in all_files:
        file_full_path = os.path.join(source_folder, file)
        # 跳过子目录，只处理文件
        if os.path.isfile(file_full_path):
            basename, ext = os.path.splitext(file)
            ext_lower = ext.lower()
            file_path_map[file] = file_full_path  # 记录完整路径

            # 根据扩展名分类
            if ext_lower in image_exts:
                image_basenames.add(basename)
            elif ext_lower == annotation_ext.lower():
                json_basenames.add(basename)

    # 找出同时存在于图像文件集和JSON文件集的基本名（即配对成功的文件）
    paired_basenames = image_basenames.intersection(json_basenames)

    print(f"在源文件夹中发现 {len(image_basenames)} 个图像文件，{len(json_basenames)} 个JSON文件。")
    print(f"找到 {len(paired_basenames)} 对匹配的文件。")

    # 开始复制配对的文件
    copied_count = 0
    for basename in paired_basenames:
        # 构建可能的文件名组合（考虑到大小写和扩展名变体）
        found_files = []
        for candidate_file in file_path_map.keys():
            candidate_basename, candidate_ext = os.path.splitext(candidate_file)
            if candidate_basename == basename:
                found_files.append(candidate_file)

        # 复制这一对文件
        for file_to_copy in found_files:
            source_path = file_path_map[file_to_copy]
            destination_path = os.path.join(target_folder, file_to_copy)
            shutil.copy2(source_path, destination_path)
            # print(f"已复制: {file_to_copy}") # 如需详细日志可取消注释
        copied_count += len(found_files)

    print(f"操作完成！已将 {len(paired_basenames)} 对文件（共 {copied_count} 个文件）复制到目标文件夹: {target_folder}")


# === 使用示例 ===
if __name__ == "__main__":
    # 请修改为您的实际路径
    source_directory = r"E:\data\已标注\正面的四个侧面_缺陷检测"  # 替换为你的源文件夹路径
    target_directory = r"E:\data\已标注\正面的四个侧面_缺陷检测_"  # 替换为你的目标文件夹路径

    # 执行复制
    copy_paired_files(source_directory, target_directory)