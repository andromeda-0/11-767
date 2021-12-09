- Offline Down-sampling.

```
/usr/bin/python3 classification.py --test --data_root /home/zongyuez/data/FaceMask_224 --gpu_id 0 --num_workers 0 --batch 1 --model MobileNetV3Small_All --epoch 2 --load MobileNetV3Small_All_class_b=64lr=0.001_r224
100%|████████████████████████████████████████████████████████████████████████████████████| 286/286 [01:05<00:00,  4.38it/s]
epoch:  2 Test Loss:  0.00011 Overall Accuracy:  1.00000 Latency:  28.33372 [ms] Mean Accuracy:  1.00000
```

- Online Down-sampling

```
/usr/bin/python3 classification.py --test --data_root /home/zongyuez/data/FaceMask --gpu_id 0 --num_workers 0 --batch 1
--model MobileNetV3Small_All --epoch 2 --load MobileNetV3Small_All_class_b=64lr=0.001_r224 --resize 224
100%|████████████████████████████████████████████████████████████████████████████████████|
286/286 [01:44<00:00, 2.74it/s]
epoch:  2 Test Loss:  0.00010 Overall Accuracy:  1.00000 Latency:  29.41479 [ms] Mean Accuracy:  1.00000
```

- No Down-sampling
```
/usr/bin/python3 classification.py --test --data_root /home/zongyuez/data/FaceMask --gpu_id 0 --num_workers 0 --batch 1 --model MobileNetV3Small_All --epoch 6 --load MobileNetV3Small_All_class_b=16lr=0.001_
100%|████████████████████████████████████████████████████████████████████████████████████| 286/286 [02:17<00:00,  2.08it/s]
epoch:  6 Test Loss:  0.00019 Overall Accuracy:  1.00000 Latency:  36.83902 [ms] Mean Accuracy:  1.00000
```

- Large MobileNet Model
```
/usr/bin/python3 classification.py --test --data_root /home/zongyuez/data/FaceMask --gpu_id 0 --num_workers 0 --batch 1 --model MobileNetV3Large_All --epoch 8 --load MobileNetV3Large_All_class_b=16lr=0.001_
100%|████████████████████████████████████████████████████████████████████████████████████| 286/286 [05:09<00:00,  1.08s/it]
epoch:  8 Test Loss:  0.00000 Overall Accuracy:  1.00000 Latency:  145.61503 [ms] Mean Accuracy:  1.00000
```

- Downsample to 64X64
```
/usr/bin/python3 classification.py --test --data_root /home/zongyuez/data/FaceMask --gpu_id 0 --num_workers 0 --batch 1 --model MobileNetV3Small_All --epoch 8 --resize 64 --load MobileNetV3Small_All_class_b=64lr=0.001_r64
100%|████████████████████████████████████████████████████████████████████████████████████| 286/286 [01:32<00:00,  3.08it/s]
epoch:  8 Test Loss:  0.00061 Overall Accuracy:  1.00000 Latency:  30.90408 [ms] Mean Accuracy:  1.00000
```

- Downsample to 32x32
```
/usr/bin/python3 classification.py --test --data_root /ho
me/zongyuez/data/FaceMask --gpu_id 0 --num_workers 0 --batch 1 --model MobileNetV3Small_All --epoch 8 --resize 32 --load MobileNetV3Small_All_class_b=64lr=0.001_r32
100%|████████████████████████████████████████████████████████████████████████████████████| 286/286 [01:31<00:00,  3.13it/s]
epoch:  8 Test Loss:  0.00215 Overall Accuracy:  1.00000 Latency:  31.84461 [ms] Mean Accuracy:  1.00000
```

- Downsample to 32x32, online, CPU
```
/usr/bin/python3 classification.py --test --data_root /ho
me/zongyuez/data/FaceMask --gpu_id -1 --num_workers 0 --batch 1 --model MobileNetV3Small_All --epoch 8 --resize 32 --load MobileNetV3Small_All_class_b=64lr=0.001_r32
100%|████████████████████████████████████████████████████████████████████████████████████| 286/286 [01:38<00:00,  2.90it/s]
epoch:  8 Test Loss:  0.00215 Overall Accuracy:  1.00000 Latency:  155.15429 [ms] Mean Accuracy:  1.00000
```

- Downsample to 32x32, offline, CPU
```
/usr/bin/python3 classification.py --test --data_root /ho
me/zongyuez/data/FaceMask_32 --gpu_id -1 --num_workers 0 --batch 1 --model MobileNetV3Small_All --epoch 8 --resize -1 --load MobileNetV3Small_All_class_b=64lr=0.001_r32
100%|████████████████████████████████████████████████████████████████████████████████████| 286/286 [01:06<00:00,  4.27it/s]
epoch:  8 Test Loss:  0.00216 Overall Accuracy:  1.00000 Latency:  140.86811 [ms] Mean Accuracy:  1.00000
```
