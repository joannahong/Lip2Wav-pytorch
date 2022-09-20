# Lip Dataset Tools

该文件夹提供用于整理数据集的相关工具脚本

1. 下载数据集
```shell
conda create -n dsTools python=3.7
conda activate dsTools 
pip install youtube_dl
apt-get install ffmpeg

sh download_speaker.sh Datasets/chem
```
2. `separate_audio.py`：从视频中提取音频文件，并保存为 wav 文件。
3. `save_melspec.py`：将音频文件转换为melspec，并保存于 npz 文件中。（2500条数据约需40分钟）
4. `save_landmark.py`：将视频文件导出为Landmark，并保存于 npz 文件中。（2500条数据约需10小时）
5. `save_json.py`（已废弃）：将视频文件导出为Landmark，并保存于 json 文件中。

```python
landmark:
{
    landmark: (landmark_pad, 478, 3)
    detect_info: landmark_pad *  [score, width, height, angle,  # 评分, 宽度, 高度, 方框旋转角度
                                  center_x, center_y,  #  中心点
                                  p1x, p1y, p2x, p2y,  #  左上, 右上
                                  p3x, p3y, p4x, p4y]  #  右下, 左下
    base_info: [frame_count, width, height]
}

melspec:
{
    melspec: (num_mels, audio_pad)
    audio: (audio_time * sr, )
}
```