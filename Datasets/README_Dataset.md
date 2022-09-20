# 数据集说明



```shell
chem/
    intervals/
    	zero.txt						# 全部为0的文件，在dataset中排除
		Videos/
		Audios/
        npz/
        	Melspce/					# 音频和频谱 npz
        	Landmark/				    # 识别的 Landmark的 npz
        	NormLandmark/			    # 正规化的 Landmark
        	RecogFaceLandmark/		    # npz分离的 Landmark
        	RecogFaceLandmark1/		    # 正规化+npz分离的 Landmark
        
```



### 数据集预处理
```shell
cd /home/g20tka12/Developer/LipLandmark2Wav
conda activate faceDect

nohup python preprocess.py --hp $HPNAME > ~/tmp_preprocessor.out &

cd /home/g20tka12/Developer/Datasets
scp -r ./intervals g20tka12@192.168.2.213:/home/g20tka12/Developer/Datasets/chem/intervals
```

### 数据集目录示例
```
├── chem
│   ├── test.txt
│   ├── train.txt
│   ├── val.txt
│   ├── intervals
|   |   ├── Audios
|   |   |   ├── _7s29Q76st0
|   |   |   |   ├── cut0.wav
|   |   |   |   ├── cut1.wav
|   |   |   |   ├── ......
|   |   |   ├── _56-KofIBng 
|   |   |   ├── ......
|   |   ├── Videos
|   |   |   ├── _7s29Q76st0
|   |   |   |   ├── cut0.mp4
|   |   |   |   ├── cut1.mp4
|   |   |   |   ├── ......
|   |   |   ├── _56-KofIBng 
|   |   |   ├── ......
|   |   ├── npz 
|   |   |   ├── Landmark
|   |   |   |   ├── _7s29Q76st0
|   |   |   |   |   ├── cut0.npz
|   |   |   |   |   |   ├── landmark        # shape(landmark_pad, 478, 3)
|   |   |   |   |   |   ├── detect_info     # landmark_pad * [score, xmin, ymin, width, height]
|   |   |   |   |   |   ├── base_info       # [frame_count, width, height]
|   |   |   |   |   ├── cut1.npz
|   |   |   |   |   ├── ......
|   |   |   |   ├── _56-KofIBng 
|   |   |   |   ├── ......
|   |   |   ├── Melspec
|   |   |   |   ├── _7s29Q76st0
|   |   |   |   |   ├── cut0.npz
|   |   |   |   |   |   ├── melspec          # shape(audio_pad, num_mels)
|   |   |   |   |   |   ├── audio            # shape(audio_time * sr, )
|   |   |   |   |   ├── cut1.npz
|   |   |   |   |   ├── ......
|   |   |   |   ├── _56-KofIBng 
|   |   |   |   ├── ......
│   ├── preprocessed
│   ├── videos
```
