import os
import shutil
from pathlib import Path


def move_all_files(source_folder, target_folder, conflict_resolution='rename'):
    """
    将源文件夹中所有子文件夹内的文件移动到目标文件夹

    参数:
        source_folder: 源文件夹路径
        target_folder: 目标文件夹路径
        conflict_resolution: 冲突处理方式 ('overwrite'覆盖, 'rename'重命名, 'skip'跳过)
    """
    # 确保目标文件夹存在
    Path(target_folder).mkdir(parents=True, exist_ok=True)

    moved_count = 0
    skipped_count = 0

    # 遍历源文件夹及其所有子文件夹
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            # 源文件完整路径
            source_path = os.path.join(root, file)

            # 目标文件路径
            target_path = os.path.join(target_folder, file)

            try:
                # 如果目标文件已存在，根据策略处理
                if os.path.exists(target_path):
                    if conflict_resolution == 'overwrite':
                        # 覆盖现有文件
                        shutil.copy(source_path, target_path)
                        print(f"覆盖移动: {source_path} -> {target_path}")

                    elif conflict_resolution == 'rename':
                        # 重命名文件
                        base, ext = os.path.splitext(file)
                        counter = 1
                        new_target_path = target_path
                        while os.path.exists(new_target_path):
                            new_target_path = os.path.join(target_folder, f"{base}_{counter}{ext}")
                            counter += 1
                        shutil.copy(source_path, new_target_path)
                        print(f"重命名移动: {source_path} -> {new_target_path}")

                    elif conflict_resolution == 'skip':
                        print(f"跳过文件(已存在): {source_path}")
                        skipped_count += 1
                        continue
                else:
                    # 直接移动文件
                    shutil.copy(source_path, target_path)
                    print(f"移动: {source_path} -> {target_path}")

                moved_count += 1

            except Exception as e:
                print(f"错误: 移动 {source_path} 时出错 - {str(e)}")

    print(f"\n操作完成! 成功移动 {moved_count} 个文件, 跳过 {skipped_count} 个文件.")


def move_files_with_structure(source_folder, target_folder):
    """
    移动文件但保留原始文件夹结构
    """
    # 确保目标文件夹存在
    Path(target_folder).mkdir(parents=True, exist_ok=True)

    moved_count = 0

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            source_path = os.path.join(root, file)

            # 计算相对于源文件夹的相对路径
            relative_path = os.path.relpath(source_path, source_folder)
            target_path = os.path.join(target_folder, relative_path)

            # 确保目标目录存在
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            try:
                shutil.copy(source_path, target_path)
                print(f"移动: {source_path} -> {target_path}")
                moved_count += 1
            except Exception as e:
                print(f"错误: 移动 {source_path} 时出错 - {str(e)}")

    print(f"\n操作完成! 成功移动 {moved_count} 个文件.")


if __name__ == "__main__":
    # 设置你的文件夹路径
    source_folder = r"E:\data\新建文件夹 (2)\气孔"  # 替换为你的源文件夹路径
    target_folder = r"E:\data\气孔"  # 替换为你的目标文件夹路径

    # 使用方法1: 将所有文件移动到同一文件夹
    print("=== 方法1: 合并所有文件到同一文件夹 ===")
    move_all_files(source_folder, target_folder, conflict_resolution='rename')

    # 使用方法2: 移动文件并保留文件夹结构
    # print("=== 方法2: 移动文件并保留文件夹结构 ===")
    # move_files_with_structure(source_folder, target_folder)