import os
import shutil

import cv2
from moviepy.editor import ImageSequenceClip, AudioFileClip
import numpy as np
from scipy.io.wavfile import write as write_wav

import evalatar

TEST_ROOT_DIR = "temp_sync_test_data"

def create_test_video(
    output_path: str,
    visual_delay_secs: float = 0.0,
    duration_secs: int = 3,
    fps: int = 25
):
    """
    创建一个包含视觉和音频同步信号的测试视频。
    视觉信号：一个在屏幕中央闪烁的白色方块。
    音频信号：与方块闪烁同步的“哔”声。
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    num_frames = duration_secs * fps
    beep_duration_samples = int(0.1 * 44100)
    
    # 1. 生成音频：在每秒开始时有一个哔声
    samplerate = 44100
    audio_samples = np.zeros(duration_secs * samplerate, dtype=np.int16)
    t = np.linspace(0., 0.1, beep_duration_samples, endpoint=False)
    beep = (np.sin(2. * np.pi * 440. * t) * 32767).astype(np.int16)
    for i in range(duration_secs):
        start_sample = i * samplerate
        audio_samples[start_sample : start_sample + beep_duration_samples] = beep
    
    audio_path = os.path.join(TEST_ROOT_DIR, f"temp_audio_{visual_delay_secs}.wav")
    write_wav(audio_path, samplerate, audio_samples)

    # 2. 生成视频帧：在每秒开始时闪烁一个白色方块
    frames = []
    size = (224, 224)
    for i in range(num_frames):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        time_sec = i / fps
        
        # 应用延迟
        time_sec_delayed = time_sec - visual_delay_secs
        
        # 在每秒的前0.1秒显示白色方块
        if time_sec_delayed >= 0 and time_sec_delayed % 1.0 < 0.1:
            cv2.rectangle(frame, (80, 80), (144, 144), (255, 255, 255), -1)
        
        # MoviePy 需要 RGB 格式
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 3. 合成音视频
    video_clip = ImageSequenceClip(frames, fps=fps)
    audio_clip = AudioFileClip(audio_path)
    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(output_path, codec='libx264', logger=None)
    print(f"  - 已创建测试视频: {output_path}")

def main():
    """主执行函数"""
    print("--- 1. 生成 Sync 测试视频 ---")
    synced_video_path = os.path.join(TEST_ROOT_DIR, "synced.mp4")
    desynced_video_path = os.path.join(TEST_ROOT_DIR, "desynced.mp4")
    
    create_test_video(synced_video_path, visual_delay_secs=0.0)
    create_test_video(desynced_video_path, visual_delay_secs=0.5) # 视觉延迟0.5秒

    try:
        print("\n--- 2. 开始运行 SyncNet 测试 ---")
        
        print("\n[测试 1] 评估同步视频...")
        sync_c_synced = evalatar.calculate_sync_c(synced_video_path)
        sync_d_synced = evalatar.calculate_sync_d(synced_video_path)
        print(f"-> Sync-C (置信度): {sync_c_synced:.4f} (越高越好)")
        print(f"-> Sync-D (距离):   {sync_d_synced:.4f} (越低越好)")
        
        print("\n[测试 2] 评估不同步视频...")
        sync_c_desynced = evalatar.calculate_sync_c(desynced_video_path)
        sync_d_desynced = evalatar.calculate_sync_d(desynced_video_path)
        print(f"-> Sync-C (置信度): {sync_c_desynced:.4f} (越高越好)")
        print(f"-> Sync-D (距离):   {sync_d_desynced:.4f} (越低越好)")

        print("\n--- 3. 验证测试结果 ---")
        # 由于测试视频不是真人说话，我们只做一个相对比较
        # 我们期望同步视频的置信度更高，距离更低
        assert sync_c_synced > sync_c_desynced, "同步视频的置信度应该更高"
        assert sync_d_synced < sync_d_desynced, "同步视频的距离应该更低"
        print("-> 结果: 成功！同步视频的分数优于不同步视频，符合预期。")

    except Exception as e:
        print(f"测试失败: {e}")
    finally:
        print("\n--- 4. 清理测试环境 ---")
        if os.path.exists(TEST_ROOT_DIR):
            shutil.rmtree(TEST_ROOT_DIR)
            print(f"  - 已删除临时目录: {TEST_ROOT_DIR}")

if __name__ == "__main__":
    main()