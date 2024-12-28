import numpy as np
from PIL import Image  
from numba import jit
import time
from datetime import timedelta
import matplotlib.cm as cm  

# Buddhabrot 計算関数
@jit(nopython=True)
def compute_buddhabrot(width, height, max_iter, samples):
    image = np.zeros((height, width), dtype=np.uint32)
    scale_x = 3.5 / width
    scale_y = 2.0 / height
    
    for _ in range(samples):
        x = np.random.uniform(-2.5, 1.5)
        y = np.random.uniform(-1.0, 1.0)
        c = complex(x, y)
        z = 0 + 0j
        trajectory = []
        
        for i in range(max_iter):
            z = z * z + c
            if abs(z) > 2.0:
                break
            trajectory.append((z.real, z.imag))
        else:
            continue
            
        for zx, zy in trajectory:
            px = int((zx + 2.5) / scale_x)
            py = int((zy + 1.0) / scale_y)
            if 0 <= px < width and 0 <= py < height:
                image[py, px] += 1
                
    return image

# スライスを保存する関数
def save_buddhabrot_pieces(image, width, height, num_slices, filename_base="buddhabrot_piece"):
    piece_height = height // num_slices  # 各スライスの高さを計算
    piece_files = []  # 保存したファイルのリスト
    for i in range(num_slices):
        piece = image[i * piece_height:(i + 1) * piece_height, :]
        piece_normalized = np.log1p(piece) / np.log1p(image).max()  # 0-1に正規化
        piece_colored = (cm.inferno(piece_normalized)[:, :, :3] * 255).astype(np.uint8)  # カラーマップ適用
        output_image = Image.fromarray(piece_colored)
        filename = f"{filename_base}_{i}.png"
        output_image.save(filename)
        piece_files.append(filename)
        print(f"保存しましたのよ♡: {filename}")
    return piece_files

# スライス画像を結合する関数
def combine_pieces(piece_files, width, height):
    combined_image = Image.new("RGB", (width, height))  # カラー化
    piece_height = height // len(piece_files)  # 各スライスの高さ
    for i, file in enumerate(piece_files):
        piece_image = Image.open(file)
        combined_image.paste(piece_image, (0, i * piece_height))  # スライスを結合
    combined_image.save("buddhabrot_highres_combined.png")
    print("結合した画像を保存したのよ♡: buddhabrot_highres_combined.png")

width, height = 10000, 5000  # 画像サイズ
max_iter = 100000               # 最大反復回数
samples = 10000000            # 点の数
num_slices = 10               # 分割数

# 実行時間
print("生成中なのよ♡...")
start_time = time.time()
image = compute_buddhabrot(width, height, max_iter, samples)
end_time = time.time()
actual_time = end_time - start_time
print(f"実行時間: {timedelta(seconds=int(actual_time))}")


print("画像をスライスして保存するのよ♡...")
piece_files = save_buddhabrot_pieces(image, width, height, num_slices)

# スライスの結合
print("スライス画像を結合するのよ♡...")
combine_pieces(piece_files, width, height)
