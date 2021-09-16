Lab 1: Project Brainstorming

**===**

The goal of this lab is for you to work as a group to brainstorm project ideas. At the end of this exercise, you should have an outline for your project proposal that you can share with others to get feedback. We will share these outlines for peer feedback in the next lab.



Group name: TripleFighting

**---**

Group members present in lab today: Songhao Jia, Jiajun Bao, Zongyue Zhao

1: Ideas

**----**

Write down 3-5 project ideas your group considered (a few sentences each). Depending on your group and process, the ideas may be very different, or they may be variations on one central idea.

1. Portable face mask detector

   1. target: detect whether people wearing a face mask
   2. inference: given image, propose if anyone is in the image and if so, detect whether he/she is wearing a face mask or not.
   3. maximizing power: 
      1. pruning detection model 
      2. using a microphone to detect whether there’s a person in front of the camera; if no one closes to the camera and terminate the inference of the cv model to decrease the power consumption.
   4. performing training/updates for interactive ML:
      1. if the detector fails to detect that someone is in its front, he/she needs to press the button we gave.
      2. to update the model when connecting to the battery.

2. Automatic check-in system under Covid-19

   This idea is extended from the first one, but it moves one step further. Our system will (1) detect if a person is wearing a mask. (2) recognize how he/she is.

3. An on-device speech translation system
   1. Target: given the speech of a user in language A, our system will recognize its contents, translate it into language B, and then return the results to the user through speech.
   2. Inference: our system will perform speech recognition, translation, and text-to-speech inference on the device.
   3. Performing training/updates for interactive ML: user’s speech and text contents will be recorded and stored on the device, helping improve the language model.
   4. Maximizing Power: the system will perform the training when it is connected to power.

4. Cheater Detector
   1. Target: detect whether the carrier cheats his/her mate. 
   2. Add a GPS and network module on the board
   3. Put the board into the carrier’s bag, and collect the place the carrier usually go to during the week
   4. If the carrier goes somewhere unusual, send a message to his/her mate.



2: Narrowing

**----**

Choose two of the above ideas. For each one:

1. How would this project leverage the expertise of each member of your group?

2. In completing this project, what new things would you learn about: (a) hardware (b) efficiency in machine learning (c) I/O modalities (d) other?

3. What potential roadblocks or challenges do you foresee in this project? How might you adjust the project scope in case these aspects present insurmountable challenges?

4. How could you potentially extend the scope of this project if you had e.g. one more month?



Portable face mask detector

1. Jiajun Bao has prior knowledge in model pruning, and all of us have experience in designing and developing cv deep learning models. 
2. We will learn about 1) techniques for deploying ML models on edge devices; 2)efficiency in machine learning: compression methods like pruning and distillation and distillation, etc; 3) inference frameworks like ONNX runtime.
3. Collect sound data (raw sound input v.s. Label, some one really near the machine), manipulate the devices and modal inference dynamically.
4. Not only do mask detection, but real recognize who is wearing the mask, making it become an automatic check-in system under covid pandemic.  

An on-device speech translation system

1. Jiajun has prior experience in language technologies. 
2. We will learn about 1) compressing and deploying models on edge devices; 2) performing efficient on-device inference; 3) building on-device machine learning pipelines.

Challenges: 

1. We need to design heuristics to efficiently and accurately trigger the system. If we cannot do it, we will have our system always listening.
2. We need to collect user speech data, and personalize models all on devices. If we cannot do it, we will not update our model.

Extension:

1. We will extend the translation model to cover more language pairs, or train and test multilingual translation models.



3: Outline

**----**

Choose one of the ideas you've considered, and outline a project proposal for that idea. This outline will be shared with other groups next class (Tuesday) to get feedback.

Your outline should include:

1. Motivation

​    Currently, CMU mandatorily demands students to wear mask before entering building, but there is no device to examine and detect it. We want to build a power-efficient mask detector on the embed system to overcome the issue. 

2. Hypotheses (key ideas)
   1. Deploy a mask detector on embed system, and the power consumption of the camera as well as the inference of cv modal would cost a lot. 
   2. There would  be a long time that no one would comes across the camera. Our system would stop using camera and inference the cv modal. 
   3. We use a microphone and an audio deep learning module to detect if there is anyone near to the modal; if so, start to infer the modal and camera; else, close them.
   4. We can automatically collect the data for the microphone: we open the microphone and camera all the time, and deploy a human detection in the system. It could collect if there’s a person neat the system and its corresponding sound all the time. 

3. How you will test those hypotheses: datasets, baselines, ablations, and other experiments or analyses.
   1. Dataset: for initialization, we will first train on publicly available mask datasets. Then, we will do (online) transfer learning to adapt to CMU students.
   2. Baseline: mobilenet?
   3. Ablations: distillation, pruning, FP16, etc.
   4. We will hold a separate test set to evaluate models’ performance. The performance is measured by the recall, precision, and accuracy of the model.

4. I/O: What are the inputs and output modalities? What existing tools will you use to convert device inputs (that are not a core part of your project) to a format readable by the model, and vice versa?

5. Input/output modalities:
   1. The input is in speech (sound near the system) and vision (image of the user). We will capture the image with a camera video stream, and the sound with a microphone audio stream.
   2. The output is in speech (e.g. a speaker telling the user “you are not wearing a mask”) or vision (a light on or off indicating if the user is wearing a mask).
   3. Alternatively, treat this problem as semantic segmentation: output bounding box and/or pixel-wise mask/human segmentation.
   4. IO Tools: Opencv, libportaudio2, scipy.

6. Hardware, including any peripherals required, and reasoning for why that hardware was chosen for this project. (This is where you will request additional hardware and/or peripherals for your project!)
   1. Potentially, an IR sensor if it requires less power than the microphone.

7. Potential challenges, and how you might adjust the project to adapt to those challenges.
   1. Model size too large to deploy on-device.
   2. Transfer learning won’t converge.

8. Potential extensions to the project.
   1. Binary classification problem -> human (with mask) identification.
   2. Semantic / instance segmentation.
