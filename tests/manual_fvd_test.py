import os
import shutil
import numpy as np
import cv2
import evalatar

TEST_ROOT_DIR = "temp_fvd_test_data"
REAL_DIR = os.path.join(TEST_ROOT_DIR, "real")
GEN_DIR = os.path.join(TEST_ROOT_DIR, "generated")

def create_dummy_video(path, color, frames=16, fps=10):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    size = (224, 224)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    frame = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    for _ in range(frames):
        writer.write(frame)
    writer.release()
    print(f"  - 已创建虚拟视频: {path}")

def main():
    print("--- 1. 生成 FVD 测试视频 ---")
    # 生成两个完全相同的黑色视频
    create_dummy_video(os.path.join(REAL_DIR, "black1.mp4"), color=(0, 0, 0))
    create_dummy_video(os.path.join(GEN_DIR, "black2.mp4"), color=(0, 0, 0))
    # 生成一个不同的红色视频
    create_dummy_video(os.path.join(GEN_DIR, "red.mp4"), color=(0, 0, 255))

    try:
        print("\n--- 2. 开始运行 FVD 测试 ---")
        
        # 测试1：比较相同的视频
        print("\n[测试 1] 比较相同的视频 (黑色 vs 黑色)")
        fvd_identical = evalatar.calculate_fvd(
            os.path.join(REAL_DIR, "*.mp4"),
            os.path.join(GEN_DIR, "black2.mp4"),
            num_frames=16
        )
        print(f"-> FVD 分数 (相同): {fvd_identical}")
        assert fvd_identical < 1e-4, "相同视频的FVD分数应该接近0"
        print("-> 结果: 成功！")

        # 测试2：比较不同的视频
        print("\n[测试 2] 比较不同的视频 (黑色 vs 全部生成的视频)")
        fvd_different = evalatar.calculate_fvd(
            os.path.join(REAL_DIR, "*.mp4"),
            os.path.join(GEN_DIR, "*.mp4"),
            num_frames=16
        )
        print(f"-> FVD 分数 (不同): {fvd_different}")
        assert fvd_different > 0, "不同视频的FVD分数应该是一个正数"
        print("-> 结果: 成功！")

    finally:
        print("\n--- 3. 清理测试环境 ---")
        if os.path.exists(TEST_ROOT_DIR):
            shutil.rmtree(TEST_ROOT_DIR)
            print(f"  - 已删除临时目录: {TEST_ROOT_DIR}")

if __name__ == "__main__":
    main()