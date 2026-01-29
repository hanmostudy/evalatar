import os
import shutil
import numpy as np
import cv2
from moviepy.editor import ImageSequenceClip

# 导入你自己的模块
import evalatar

TEST_ROOT_DIR = "temp_iqa_test_data"
HQ_VIDEO_DIR = os.path.join(TEST_ROOT_DIR, "hq_videos")
LQ_VIDEO_DIR = os.path.join(TEST_ROOT_DIR, "lq_videos")

def create_test_videos(
    duration_secs: int = 2,
    fps: int = 10,
    size=(256, 256)
):
    """
    创建两个视频：一个高质量的，一个经过模糊和压缩的低质量的。
    视频内容是一个移动的彩色方块。
    """
    os.makedirs(HQ_VIDEO_DIR, exist_ok=True)
    os.makedirs(LQ_VIDEO_DIR, exist_ok=True)
    
    hq_frames = []
    lq_frames = []
    num_frames = duration_secs * fps

    for i in range(num_frames):
        # 1. 创建一帧高质量的画面
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8) + 50 # 深灰色背景
        
        # 计算方块位置
        x_pos = int((i / num_frames) * (size[0] - 50))
        y_pos = int(np.sin((i / num_frames) * np.pi) * (size[1] - 50))
        
        # 绘制一个移动的彩色方块
        color = (100 + i % 155, 200 - i % 100, 50 + i % 200) # BGR color
        cv2.rectangle(frame, (x_pos, y_pos), (x_pos + 50, y_pos + 50), color, -1)
        
        # 2. 创建一帧对应的低质量画面
        # a. 施加高斯模糊
        frame_blurred = cv2.GaussianBlur(frame, (15, 15), 0)
        
        # b. 模拟JPEG压缩伪影
        _, buffer = cv2.imencode('.jpg', frame_blurred, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
        frame_lq = cv2.imdecode(buffer, 1)

        # MoviePy 需要 RGB 格式
        hq_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        lq_frames.append(cv2.cvtColor(frame_lq, cv2.COLOR_BGR2RGB))
        
    # 3. 保存视频
    hq_clip = ImageSequenceClip(hq_frames, fps=fps)
    lq_clip = ImageSequenceClip(lq_frames, fps=fps)
    
    hq_path = os.path.join(HQ_VIDEO_DIR, "high_quality.mp4")
    lq_path = os.path.join(LQ_VIDEO_DIR, "low_quality.mp4")
    
    hq_clip.write_videofile(hq_path, codec='libx264', logger=None, audio=False)
    lq_clip.write_videofile(lq_path, codec='libx264', logger=None, audio=False)
    
    print(f"  - 已创建高质量视频: {hq_path}")
    print(f"  - 已创建低质量视频: {lq_path}")


def main():
    """主执行函数"""
    print("--- 1. 生成 IQA 测试视频 ---")
    create_test_videos()

    try:
        hq_path_pattern = os.path.join(HQ_VIDEO_DIR, "*.mp4")
        lq_path_pattern = os.path.join(LQ_VIDEO_DIR, "*.mp4")

        # --- 测试 1: 使用 BRISQUE (快速, CPU友好, 分数越低越好) ---
        print("\n--- 2. 开始使用 BRISQUE 进行测试 (分数越低越好) ---")
        
        print("\n[测试 1a] 评估高质量视频...")
        score_hq_brisque = evalatar.calculate_iqa(
            generated_videos_path=hq_path_pattern,
            metric_name='brisque',
            frame_sample_rate=5 # 每秒采5帧
        )
        
        print("\n[测试 1b] 评估低质量视频...")
        score_lq_brisque = evalatar.calculate_iqa(
            generated_videos_path=lq_path_pattern,
            metric_name='brisque',
            frame_sample_rate=5
        )
        
        print("\n[验证 BRISQUE 结果]")
        print(f"-> 高质量视频 BRISQUE 分数: {score_hq_brisque:.4f}")
        print(f"-> 低质量视频 BRISQUE 分数: {score_lq_brisque:.4f}")
        assert score_lq_brisque > score_hq_brisque, "BRISQUE: 低质量视频的分数应该更高"
        print("-> 结果: 成功！BRISQUE 分数符合预期。")

        # --- 测试 2: 使用 MANIQA (先进, 需要GPU, 分数越高越好) ---
        print("\n--- 3. 开始使用 MANIQA 进行测试 (分数越高越好) ---")
        print("(首次运行会下载模型，在CPU上会非常慢，请耐心等待...)")

        # 检查是否有可用的 GPU，如果没有则跳过 MANIQA 测试
        import torch
        if not torch.cuda.is_available():
            print("\n警告: 未检测到可用 CUDA 设备，跳过 MANIQA 测试。")
        else:
            print("\n[测试 2a] 评估高质量视频...")
            score_hq_maniqa = evalatar.calculate_iqa(
                generated_videos_path=hq_path_pattern,
                metric_name='maniqa',
                frame_sample_rate=5
            )
            
            print("\n[测试 2b] 评估低质量视频...")
            score_lq_maniqa = evalatar.calculate_iqa(
                generated_videos_path=lq_path_pattern,
                metric_name='maniqa',
                frame_sample_rate=5
            )

            print("\n[验证 MANIQA 结果]")
            print(f"-> 高质量视频 MANIQA 分数: {score_hq_maniqa:.4f}")
            print(f"-> 低质量视频 MANIQA 分数: {score_lq_maniqa:.4f}")
            assert score_hq_maniqa > score_lq_maniqa, "MANIQA: 高质量视频的分数应该更高"
            print("-> 结果: 成功！MANIQA 分数符合预期。")

    except Exception as e:
        print(f"\n测试失败: {e}")
    finally:
        print("\n--- 4. 清理测试环境 ---")
        if os.path.exists(TEST_ROOT_DIR):
            shutil.rmtree(TEST_ROOT_DIR)
            print(f"  - 已删除临时目录: {TEST_ROOT_DIR}")

if __name__ == "__main__":
    main()