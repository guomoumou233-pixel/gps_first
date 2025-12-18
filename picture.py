import os
from PIL import Image

def get_image_dimensions(image_path: str) -> str:
    """
    查看指定路径图片文件的尺寸，并以 "Width×Height" 格式返回。

    Args:
        image_path: 图片文件的完整路径。

    Returns:
        返回图片的尺寸字符串，例如 "1024x768"。
        如果文件不存在或不是有效图片，返回错误信息。
    """
    # 1. 检查文件是否存在
    if not os.path.exists(image_path):
        return f"错误: 文件不存在于路径: {image_path}"

    try:
        # 2. 使用 PIL 库打开图片
        with Image.open(image_path) as img:
            # 3. 获取图片的宽度和高度
            width, height = img.size
            
            # 4. 格式化输出为 Width×Height
            return f"{width}×{height}"

    except IOError:
        return f"错误: 无法打开或识别文件 {image_path}。请确保它是有效的图片格式。"
    except Exception as e:
        return f"发生未知错误: {e}"

# --- 主执行部分 ---
if __name__ == "__main__":
    # 替换为您指定的路径
    specified_path = "/root/mnist-clip/RS_images_2800/aGrass/a001.jpg" 
    
    # 调用函数并打印结果
    dimensions = get_image_dimensions(specified_path)
    
    print(f"检查文件: {specified_path}")
    print(f"图片尺寸为: {dimensions}")

    # 如果图片尺寸是 224x224，输出将是：
    # 图片尺寸为: 224×224