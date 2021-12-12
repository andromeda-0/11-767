- Offline Down-sampling.

```bash
/usr/bin/python3 classification.py --test --data_root /home/zongyuez/data/FaceMask_224 --gpu_id 0 --num_workers 0 --batch 1 --model MobileNetV3Small_All --epoch 2 --load MobileNetV3Small_All_class_b=64lr=0.001_r224
100%|████████████████████████████████████████████████████████████████████████████████████| 286/286 [01:05<00:00,  4.38it/s]
epoch:  2 Test Loss:  0.00011 Overall Accuracy:  1.00000 Latency:  28.33372 [ms] Mean Accuracy:  1.00000
```

- Online Down-sampling

```bash
/usr/bin/python3 classification.py --test --data_root /home/zongyuez/data/FaceMask --gpu_id 0 --num_workers 0 --batch 1
--model MobileNetV3Small_All --epoch 2 --load MobileNetV3Small_All_class_b=64lr=0.001_r224 --resize 224
100%|████████████████████████████████████████████████████████████████████████████████████|
286/286 [01:44<00:00, 2.74it/s]
epoch:  2 Test Loss:  0.00010 Overall Accuracy:  1.00000 Latency:  29.41479 [ms] Mean Accuracy:  1.00000
```

- No Down-sampling
```bash
/usr/bin/python3 classification.py --test --data_root /home/zongyuez/data/FaceMask --gpu_id 0 --num_workers 0 --batch 1 --model MobileNetV3Small_All --epoch 6 --load MobileNetV3Small_All_class_b=16lr=0.001_
100%|████████████████████████████████████████████████████████████████████████████████████| 286/286 [02:17<00:00,  2.08it/s]
epoch:  6 Test Loss:  0.00019 Overall Accuracy:  1.00000 Latency:  36.83902 [ms] Mean Accuracy:  1.00000
```

- Large MobileNet Model
```bash
/usr/bin/python3 classification.py --test --data_root /home/zongyuez/data/FaceMask --gpu_id 0 --num_workers 0 --batch 1 --model MobileNetV3Large_All --epoch 8 --load MobileNetV3Large_All_class_b=16lr=0.001_
100%|████████████████████████████████████████████████████████████████████████████████████| 286/286 [05:09<00:00,  1.08s/it]
epoch:  8 Test Loss:  0.00000 Overall Accuracy:  1.00000 Latency:  145.61503 [ms] Mean Accuracy:  1.00000
```

- Downsample to 64X64
```bash
/usr/bin/python3 classification.py --test --data_root /home/zongyuez/data/FaceMask --gpu_id 0 --num_workers 0 --batch 1 --model MobileNetV3Small_All --epoch 8 --resize 64 --load MobileNetV3Small_All_class_b=64lr=0.001_r64
100%|████████████████████████████████████████████████████████████████████████████████████| 286/286 [01:32<00:00,  3.08it/s]
epoch:  8 Test Loss:  0.00061 Overall Accuracy:  1.00000 Latency:  30.90408 [ms] Mean Accuracy:  1.00000
```

- Downsample to 32x32
```bash
/usr/bin/python3 classification.py --test --data_root /home/zongyuez/data/FaceMask --gpu_id 0 --num_workers 0 --batch 1 --model MobileNetV3Small_All --epoch 8 --resize 32 --load MobileNetV3Small_All_class_b=64lr=0.001_r32
100%|████████████████████████████████████████████████████████████████████████████████████| 286/286 [01:31<00:00,  3.13it/s]
epoch:  8 Test Loss:  0.00215 Overall Accuracy:  1.00000 Latency:  31.84461 [ms] Mean Accuracy:  1.00000
```

- Downsample to 32x32, online, CPU
```bash
/usr/bin/python3 classification.py --test --data_root /home/zongyuez/data/FaceMask --gpu_id -1 --num_workers 0 --batch 1 --model MobileNetV3Small_All --epoch 8 --resize 32 --load MobileNetV3Small_All_class_b=64lr=0.001_r32
100%|████████████████████████████████████████████████████████████████████████████████████| 286/286 [01:38<00:00,  2.90it/s]
epoch:  8 Test Loss:  0.00215 Overall Accuracy:  1.00000 Latency:  155.15429 [ms] Mean Accuracy:  1.00000
```

- Downsample to 32x32, offline, CPU
```bash
/usr/bin/python3 classification.py --test --data_root /home/zongyuez/data/FaceMask_32 --gpu_id -1 --num_workers 0 --batch 1 --model MobileNetV3Small_All --epoch 8 --resize -1 --load MobileNetV3Small_All_class_b=64lr=0.001_r32
100%|████████████████████████████████████████████████████████████████████████████████████| 286/286 [01:06<00:00,  4.27it/s]
epoch:  8 Test Loss:  0.00216 Overall Accuracy:  1.00000 Latency:  140.86811 [ms] Mean Accuracy:  1.00000
```

----

Overall Latency - Vision:

- GPU, Larger Model, 1024x1024
```bash
/usr/bin/python3 caller_classification.py --vision_device cuda:0 --vision_weights_name MobileNetV3Large_All_class_b=16lr=0.001_ --vision_model_name MobileNetV3Large_All --image_root /home/zongyuez/data/FaceMask
Image:  /home/zongyuez/data/FaceMask/test/NMFD/60086.png True Label:  NMFD Predicted Label:  NMFD
Time Used: 67468.7
```

- GPU, Simplified Model, 1024x1024
```bash
/usr/bin/python3 caller_classification.py --vision_device cuda:0 --vision_weights_name MobileNetV3Small_All_class_b=16lr=0.001_ --vision_model_name MobileNetV3Small_All --image_root /home/zongyuez/data/FaceMask --epoch 6
Image:  /home/zongyuez/data/FaceMask/test/NMFD/60095.png True Label:  NMFD Predicted Label:  NMFD
Time Used: 55251.6
```

- GPU, Simplified Model, 224X224, Offline
```bash
/usr/bin/python3 caller_classification.py --vision_device cuda:0 --vision_weights_name MobileNetV3Small_All_class_b=64lr=0.001_r224 --vision_model_name MobileNetV3Small_All --image_root /home/zongyuez/data/FaceMask_224 --epoch 2
Image:  /home/zongyuez/data/FaceMask_224/test/NMFD/60007.png True Label:  NMFD Predicted Label:  NMFD
Time Used: 55917.1
```

- GPU, Simplified Model, 32x32, Offline
```bash
/usr/bin/python3 caller_classification.py --vision_device cuda:0 --vision_weights_name MobileNetV3Small_All_class_b=64lr=0.001_r32 --vision_model_name MobileNetV3Small_All --image_root /home/zongyuez/data/FaceMask_32 --epoch 8
Image:  /home/zongyuez/data/FaceMask_32/test/NMFD/60046.png True Label:  NMFD Predicted Label:  NMFD
Time Used: 51255.7
```

- GPU, Simplified Model, 224X224, Online
```bash
/usr/bin/python3 caller_classification.py --vision_device cuda:0 --vision_weights_name MobileNetV3Small_All_class_b=64lr=0.001_r224 --vision_model_name MobileNetV3Small_All --image_root /home/zongyuez/data/FaceMask --epoch 2 --resize 224
Image:  /home/zongyuez/data/FaceMask/test/IMFD/60016_Mask_Mouth_Chin.jpg True Label:  IMFD Predicted Label:  IMFD
Time Used: 50822.3
```

- GPU, Simplified Model, 32x32, Online
```bash
/usr/bin/python3 caller_classification.py --vision_device cuda:0 --vision_weights_name MobileNetV3Small_All_class_b=64lr=0.001_r32 --vision_model_name MobileNetV3Small_All --image_root /home/zongyuez/data/FaceMask --epoch 8 --resize 32
Image:  /home/zongyuez/data/FaceMask/test/CMFD/60092_Mask.jpg True Label:  CMFD Predicted Label:  CMFD
Time Used: 52168.6
```

- CPU, Simplified Model, 224x224, Online
```bash
/usr/bin/python3 caller_classification.py --vision_device cpu --vision_weights_name MobileNetV3Small_All_class_b=64lr=0.001_r224 --resize 224 --vision_model_name MobileNetV3Small_All --image_root /home/zongyuez/data/FaceMask --epoch 2
Image:  /home/zongyuez/data/FaceMask/test/CMFD/60009_Mask.jpg True Label:  CMFD Predicted Label:  CMFD
Time Used: 2961.7
```

- CPU, Simplified Model, 32x32, Online
```bash
/usr/bin/python3 caller_classification.py --vision_device cpu --vision_weights_name MobileNetV3Small_All_class_b=64lr=0.001_r32 --resize 32 --vision_model_name MobileNetV3Small_All --image_root /home/zongyuez/data/FaceMask --epoch 9
Image:  /home/zongyuez/data/FaceMask/test/CMFD/60037_Mask.jpg True Label:  CMFD Predicted Label:  CMFD
Time Used: 268.7
```

- CPU, Simplified Model, 32x32, Offline
```bash
/usr/bin/python3 caller_classification.py --vision_device cpu --vision_weights_name MobileNetV3Small_All_class_b=64lr=0.001_r32 --vision_model_name MobileNetV3Small_All --image_root /home/zongyuez/data/FaceMask_32 --epoch 9
Image:  /home/zongyuez/data/FaceMask_32/test/CMFD/60042_Mask.jpg True Label:  CMFD Predicted Label:  CMFD
Time Used: 179.2
```

- CPU, Simplified Model, 32x32, Hooked Directly to Camera
```bash
Start Loading
Loading Complete
GST_ARGUS: Creating output stream
CONSUMER: Waiting until producer is connected...
GST_ARGUS: Available Sensor modes :
GST_ARGUS: 3264 x 2464 FR = 21.000000 fps Duration = 47619048 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;

GST_ARGUS: 3264 x 1848 FR = 28.000001 fps Duration = 35714284 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;

GST_ARGUS: 1920 x 1080 FR = 29.999999 fps Duration = 33333334 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;

GST_ARGUS: 1640 x 1232 FR = 29.999999 fps Duration = 33333334 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;

GST_ARGUS: 1280 x 720 FR = 59.999999 fps Duration = 16666667 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;

GST_ARGUS: 1280 x 720 FR = 120.000005 fps Duration = 8333333 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;

GST_ARGUS: Running with following settings:
   Camera index = 0 
   Camera mode  = 5 
   Output Stream W = 1280 H = 720 
   seconds to Run    = 0 
   Frame Rate = 120.000005 
GST_ARGUS: Setup Complete, Starting captures for 0 seconds
GST_ARGUS: Starting repeat capture requests.
CONSUMER: Producer has connected; continuing.
[ WARN:0] global /tmp/pip-req-build-7o20lxtf/opencv/modules/videoio/src/cap_gstreamer.cpp (1081) open OpenCV | GStreamer warning: Cannot query video position: status=0, value=-1, duration=-1
GST_ARGUS: Cleaning up
CONSUMER: Done Success
GST_ARGUS: Done Success
Predicted Label:  NMFD
Time Used: 154.5
```
