import os
import shutil
import numpy as np
import cv2  # opencv-python

# 导入你自己的模块
import evalatar

# --- 配置区 ---
# 定义我们将要创建的临时测试文件夹的名称
TEST_ROOT_DIR = "temp_test_data"
REAL_VIDEOS_DIR = os.path.join(TEST_ROOT_DIR, "real_videos")
GEN_VIDEOS_DIR = os.path.join(TEST_ROOT_DIR, "generated_videos")

# --- 辅助函数：用于创建虚拟视频 ---

def create_dummy_video(path: str, color: tuple, size=(64, 64), fps=10, duration_secs=1):
    """
    创建一个指定颜色的简单MP4视频文件。
    :param path: 视频保存的完整路径 (包括文件名.mp4)。
    :param color: BGR 颜色元组, 例如 (0, 0, 0) 是黑色。
    :param size: 视频分辨率 (宽, 高)。
    :param fps: 视频帧率。
    :param duration_secs: 视频时长（秒）。
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    
    num_frames = duration_secs * fps
    
    # 创建一个纯色图像帧
    frame = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    
    for _ in range(num_frames):
        writer.write(frame)
        
    writer.release()
    print(f"  - 已创建虚拟视频: {path}")

# --- 主函数：执行所有操作 ---

def main():
    """
    主执行函数：搭建环境 -> 生成数据 -> 运行测试 -> 清理环境
    """
    print("--- 1. 搭建测试环境 ---")
    # 创建所有需要的目录，如果已存在也不会报错
    os.makedirs(REAL_VIDEOS_DIR, exist_ok=True)
    os.makedirs(GEN_VIDEOS_DIR, exist_ok=True)
    print(f"  - 已创建目录: {TEST_ROOT_DIR}")

    # 使用 try...finally 结构来确保测试结束后，临时文件一定会被删除
    # 即使测试中途出错，finally 块里的代码也会执行
    try:
        print("\n--- 2. 生成测试视频 ---")
        # 生成用于“相同”测试的视频 (两个黑色的视频)
        create_dummy_video(os.path.join(REAL_VIDEOS_DIR, "black_video.mp4"), color=(0, 0, 0))
        create_dummy_video(os.path.join(GEN_VIDEOS_DIR, "black_video_copy.mp4"), color=(0, 0, 0))
        
        # 生成用于“不同”测试的视频 (一个白色的视频)
        create_dummy_video(os.path.join(GEN_VIDEOS_DIR, "white_video.mp4"), color=(255, 255, 255))

        print("\n--- 3. 开始运行测试 ---")

        # --- 测试 1: 比较内容相同的视频 ---
        print("\n[测试 1] 比较相同的视频 (黑色 vs 黑色)")
        real_path_1 = os.path.join(REAL_VIDEOS_DIR, "*.mp4")
        gen_path_1 = os.path.join(GEN_VIDEOS_DIR, "black_video_copy.mp4")
        
        fid_score_1 = evalatar.calculate_fid(real_path_1, gen_path_1, batch_size=1, dims=2048)
        print(f"-> FID 分数: {fid_score_1}")
        
        if fid_score_1 < 1e-5:
            print("-> 结果: 成功！分数非常接近 0，符合预期。")
        else:
            print("-> 结果: 失败！分数没有接近 0。")

        # --- 测试 2: 比较内容不同的视频 ---
        print("\n[测试 2] 比较不同的视频 (黑色 vs 全部生成的视频)")
        real_path_2 = os.path.join(REAL_VIDEOS_DIR, "*.mp4")
        gen_path_2 = os.path.join(GEN_VIDEOS_DIR, "*.mp4")

        fid_score_2 = evalatar.calculate_fid(real_path_2, gen_path_2, batch_size=1)
        print(f"-> FID 分数: {fid_score_2}")
        
        if fid_score_2 > 0.1:
            print("-> 结果: 成功！分数是一个明显的正数，符合预期。")
        else:
            print("-> 结果: 失败！分数太小，可能没有检测出差异。")

    finally:
        print("\n--- 4. 清理测试环境 ---")
        # 检查文件夹是否存在，然后递归删除整个文件夹及其内容
        if os.path.exists(TEST_ROOT_DIR):
            shutil.rmtree(TEST_ROOT_DIR)
            print(f"  - 已删除临时目录: {TEST_ROOT_DIR}")

# --- 脚本入口 ---
if __name__ == "__main__":
    main()