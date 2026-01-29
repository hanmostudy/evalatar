import os
import shutil
import cv2
from moviepy.editor import ImageSequenceClip

# 选择使用哪个版本
from evalatar import calculate_csim 

TEST_ROOT_DIR = "temp_csim_test_data"
GEN_VIDEO_DIR = os.path.join(TEST_ROOT_DIR, "generated_videos")
ASSETS_DIR = "test_assets"
REF_A_PATH = os.path.join(ASSETS_DIR, "person_a.jpg")
REF_B_PATH = os.path.join(ASSETS_DIR, "person_b.jpg")

def create_csim_test_video(
    output_path: str,
    base_image_path: str,
    duration_secs: int = 1,
    fps: int = 10
):
    """根据一张静态图片，创建一个静态视频。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    base_image = cv2.imread(base_image_path)
    if base_image is None:
        raise FileNotFoundError(f"无法加载图片: {base_image_path}")
    
    frame_rgb = cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB)
    frames = [frame_rgb for _ in range(duration_secs * fps)]
    
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path, codec='libx264', logger=None, audio=False)
    print(f"  - 已创建测试视频: {output_path}")

def main():
    """主执行函数"""
    # 检查基础人脸图像是否存在
    if not (os.path.exists(REF_A_PATH) and os.path.exists(REF_B_PATH)):
        print("="*60)
        print("!! 错误：测试所需的人脸图像不存在。")
        print(f"!! 请在 '{ASSETS_DIR}' 文件夹中放入两张不同人物的清晰正面照，")
        print(f"!! 并分别命名为 'person_a.jpg' 和 'person_b.jpg'。")
        print("="*60)
        return

    print("--- 1. 生成 CSIM 测试视频 ---")
    
    # 创建与参考人物 A 身份相同的视频
    gen_a_path = os.path.join(GEN_VIDEO_DIR, "video_person_a.mp4")
    create_csim_test_video(gen_a_path, REF_A_PATH)
    
    # 创建与参考人物 A 身份不同的视频
    gen_b_path = os.path.join(GEN_VIDEO_DIR, "video_person_b.mp4")
    create_csim_test_video(gen_b_path, REF_B_PATH)
    
    try:
        print("\n--- 2. 开始运行 CSIM 测试 ---")
        
        print("\n[测试 1] 评估相同身份 (参考A vs 视频A)")
        csim_same = calculate_csim(
            reference_identity_path=REF_A_PATH,
            generated_videos_path=gen_a_path,
            device = "cuda"
        )
        print(f"-> CSIM 分数 (相同身份): {csim_same:.6f}")
        assert csim_same > 0.85, f"相同身份的CSIM分数应该大于0.85，当前为{csim_same:.6f}"
        print("-> 结果: 成功！")

        print("\n[测试 2] 评估不同身份 (参考A vs 视频B)")
        csim_different = calculate_csim(
            reference_identity_path=REF_A_PATH,
            generated_videos_path=gen_b_path,
            device = "cuda"
        )
        print(f"-> CSIM 分数 (不同身份): {csim_different:.6f}")
        assert csim_different < 0.6, f"不同身份的CSIM分数应该小于0.6，当前为{csim_different:.6f}"
        print("-> 结果: 成功！")

    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n--- 3. 清理测试环境 ---")
        if os.path.exists(TEST_ROOT_DIR):
            shutil.rmtree(TEST_ROOT_DIR)
            print(f"  - 已删除临时目录: {TEST_ROOT_DIR}")

if __name__ == "__main__":
    main()