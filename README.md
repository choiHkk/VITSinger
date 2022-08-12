## Introduction
1. VITS 오픈 소스를 활용하여 Singing Voice Synthesis를 테스트합니다.
2. Voice Conversion을 기본 출력으로 사용할 목적으로 Duration Predictor는 사용하지 않습니다.
3. 데이터셋은 공개하지 않습니다. 
4. Normal Speech Dataset으로 학습할 때와 마찬가지로 Singing Speech Dataset으로 학습해도 비슷한 퀄리티의 음성을 생성할 수 있습니다.


## Docker build
1. `cd /path/to/the/VITSinger`
2. `docker build --tag VITSinger:latest .`


## Training
1. `nvidia-docker run -it --name 'VITSinger' -v /path/to/VITSinger:/home/work/VITSinger --ipc=host --privileged VITSinger:latest`
2. `cd /home/work/VITSinger`
3. `ln -s /home/work/VITSinger/data/dataset/yourdataset`
4. `python preprocess.py --data_type=korean_base --storage_path=<yourdataset_path> --target_dir=<target_dataset_path> --line_target_dir=<> --model_name=korean_base --val_len=512`
5. `python train.py -c ./configs/korean_base.json -m korean_base -d <target_dataset_path>`
6. (OPTIONAL) `tensorboard --logdir=outdir/logdir`


## Tensorboard losses
![VITSinger-tensorboard-losses1](https://user-images.githubusercontent.com/69423543/184306443-6be46dd6-c438-4e66-9606-f070d54b18dc.png)
![VITSinger-tensorboard-losses2](https://user-images.githubusercontent.com/69423543/184306450-6f72c1d9-e423-40be-a70c-02b7d3119406.png)
![VITSinger-tensorboard-losses3](https://user-images.githubusercontent.com/69423543/184306475-4808546b-58c1-4f31-b712-8d66d661710c.png)


## Tensorboard Stats
![VITSinger-tensorboard-stats](https://user-images.githubusercontent.com/69423543/184306483-a9144718-f2da-4885-8747-2d45cb6a0896.png)


## Reference
1. [HiFiSinger](https://github.com/CODEJIN/HiFiSinger)
2. [VITS](https://github.com/jaywalnut310/vits)
