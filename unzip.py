import zipfile
import os

def unzip_file(zip_path: str):
    """
    解压缩指定的 ZIP 文件，并将其内容放置在 ZIP 文件同名的新建文件夹中。

    Args:
        zip_path: ZIP 文件的完整路径。
    """
    # 1. 检查 ZIP 文件是否存在
    if not os.path.exists(zip_path):
        print(f"❌ 错误: 压缩文件不存在于路径: {zip_path}")
        return

    # 2. 确定解压缩目标路径 (同目录下同名的文件夹)
    # 获取目录路径: /root/mnist-clip
    base_dir = os.path.dirname(zip_path) 
    # 获取文件名（不含扩展名）: RS_images_2800
    file_name_without_ext = os.path.splitext(os.path.basename(zip_path))[0]
    # 构造目标文件夹路径: /root/mnist-clip/RS_images_2800
    extract_dir = os.path.join(base_dir, file_name_without_ext) 

    # 3. 创建目标文件夹（如果不存在）
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
        print(f"✅ 创建目标文件夹: {extract_dir}")
    else:
        print(f"⚠️ 目标文件夹已存在: {extract_dir}。文件将解压到现有文件夹中。")

    # 4. 执行解压缩
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            print(f"⏳ 正在解压缩 {zip_path} 到 {extract_dir}...")
            # 解压缩所有内容到目标路径
            zip_ref.extractall(extract_dir)
            print("🎉 解压缩成功!")
            
    except zipfile.BadZipFile:
        print(f"❌ 错误: {zip_path} 不是一个有效的 ZIP 文件或文件已损坏。")
    except Exception as e:
        print(f"❌ 解压缩过程中发生未知错误: {e}")

# --- 主执行部分 ---
if __name__ == "__main__":
    # 指定的 ZIP 文件路径
    ZIP_FILE_PATH = "/root/mnist-clip/RS_images_2800.zip" 
    
    unzip_file(ZIP_FILE_PATH)