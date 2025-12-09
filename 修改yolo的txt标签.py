# -*- coding: utf-8 -*-
import os, re
folder = r"D:\labels"  # <== 替换为你的文件夹路径

for file in os.listdir(folder):
    if not file.lower().endswith(".txt"): continue
    path = os.path.join(folder, file)
    with open(path, 'r', encoding="utf-8") as f:
        txt = f.read()
    # 仅替换每行开头的类别ID，避免把 10、11 中的 1 也改掉
    txt_new = re.sub(r'^(0\b)', '1', txt, flags=re.MULTILINE)
    if txt_new != txt:
        with open(path, 'w', encoding="utf-8") as f:
            f.write(txt_new)
        print("已处理:", file)