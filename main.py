import numpy as np
from PIL import Image
from numba import jit
import time
from datetime import timedelta, datetime
import matplotlib.cm as cm

# Buddhabrotフラクタルを計算する関数
@jit(nopython=True)  # 処理を高速化するためにNumbaを使用
def compute_buddhabrot(width, height, max_iter, samples):
    """
    Buddhabrotフラクタルを計算する関数
    :param width: 画像の幅（ピクセル単位）
    :param height: 画像の高さ（ピクセル単位）
    :param max_iter: フラクタルの最大反復回数
    :param samples: 計算する点の総数
    :return: フラクタルの画像データ（numpy配列）
    """
    image = np.zeros((height, width), dtype=np.uint32)  # 画像データを格納する2D配列を初期化
    scale_x = 3.5 / width  # x座標のスケール
    scale_y = 2.0 / height  # y座標のスケール

    # 指定された点数のサンプルを計算
    for _ in range(samples):
        # ランダムに座標を生成
        x = np.random.uniform(-2.5, 1.5)
        y = np.random.uniform(-1.0, 1.0)
        c = complex(x, y)  # 複素数cを作成
        z = 0 + 0j  # 初期値z=0
        trajectory = []  # 軌跡を記録するリスト

        # Mandelbrot計算を実行
        for i in range(max_iter):
            z = z * z + c  # Mandelbrot方程式
            if abs(z) > 2.0:  # 発散条件をチェック
                break
            trajectory.append((z.real, z.imag))  # 軌跡を記録
        else:
            # 発散しない場合はこのサンプルを無視
            continue

        # 発散した軌跡を画像データに反映
        for zx, zy in trajectory:
            px = int((zx + 2.5) / scale_x)  # x座標を画像上のピクセルに変換
            py = int((zy + 1.0) / scale_y)  # y座標を画像上のピクセルに変換
            if 0 <= px < width and 0 <= py < height:
                image[py, px] += 1  # ピクセル値を増加

    return image

# 画像をスライスして保存する関数
def save_buddhabrot_pieces(image, width, height, num_slices, filename_base="buddhabrot_piece"):
    """
    フラクタル画像を指定された数のスライスに分割し、保存する関数
    :param image: 生成されたフラクタル画像（numpy配列）
    :param width: 画像の幅（ピクセル単位）
    :param height: 画像の高さ（ピクセル単位）
    :param num_slices: 分割するスライスの数
    :param filename_base: 保存するファイル名の基本部分
    :return: 保存されたファイル名のリスト
    """
    piece_height = height // num_slices  # 各スライスの高さを計算
    piece_files = []  # 保存されたファイル名を格納するリスト
    for i in range(num_slices):
        piece = image[i * piece_height:(i + 1) * piece_height, :]  # スライスを切り出し
        piece_normalized = np.log1p(piece) / np.log1p(image).max()  # 値を0-1に正規化
        piece_colored = (cm.inferno(piece_normalized)[:, :, :3] * 255).astype(np.uint8)  # カラーマップを適用
        output_image = Image.fromarray(piece_colored)  # PIL Imageに変換
        filename = f"{filename_base}_{i}.png"  # ファイル名を生成
        output_image.save(filename)  # 画像を保存
        piece_files.append(filename)
        print(f"保存しましたのよ♡: {filename}")
    return piece_files

# スライス画像を結合して1つの画像にする関数
def combine_pieces(piece_files, width, height):
    """
    保存されたスライス画像を1枚の画像に結合する関数
    :param piece_files: スライス画像のファイル名リスト
    :param width: 画像の幅（ピクセル単位）
    :param height: 画像の高さ（ピクセル単位）
    """
    combined_image = Image.new("RGB", (width, height))  # 新しい空の画像を作成
    piece_height = height // len(piece_files)  # 各スライスの高さを計算
    for i, file in enumerate(piece_files):
        piece_image = Image.open(file)  # スライス画像を開く
        combined_image.paste(piece_image, (0, i * piece_height))  # スライスを結合
    combined_image.save("buddhabrot_highres_combined.png")  # 結合した画像を保存
    print("結合した画像を保存したのよ♡: buddhabrot_highres_combined.png")

# パラメータ設定
width, height = 10000, 5000  # 画像サイズ（幅×高さ）
max_iter = 100000            # 最大反復回数
samples = 60000000           # 計算する点の総数
num_slices = 10              # 画像を分割するスライスの数

# テスト実行で予想所要時間を計算
print("予想時間を計算中なのよ♡...")
test_samples = 100  # テスト用の少ないサンプル数
test_start = time.time()  # テスト開始時刻
compute_buddhabrot(100, 50, max_iter, test_samples)  # 小さい画像でテスト
test_duration = time.time() - test_start  # テスト実行時間を計算

# 総所要時間を推定
estimated_total_time = (test_duration / test_samples) * samples
estimated_end_time = datetime.now() + timedelta(seconds=estimated_total_time)
print(f"予想所要時間: {timedelta(seconds=int(estimated_total_time))}")
print(f"予想終了時刻: {estimated_end_time.strftime('%Y-%m-%d %H:%M:%S')}")

# メイン処理を実行
print("生成中なのよ♡...")
start_time = time.time()  # 開始時刻を記録
image = compute_buddhabrot(width, height, max_iter, samples)  # フラクタルを計算
end_time = time.time()  # 終了時刻を記録
actual_time = end_time - start_time
print(f"実行時間: {timedelta(seconds=int(actual_time))}")

# 画像をスライスして保存
print("画像をスライスして保存するのよ♡...")
piece_files = save_buddhabrot_pieces(image, width, height, num_slices)

# スライス画像を結合
print("スライス画像を結合するのよ♡...")
combine_pieces(piece_files, width, height)
