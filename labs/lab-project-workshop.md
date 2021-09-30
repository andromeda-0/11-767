Lab 2: Project Workshopping / Peer Feedback
===
The goal of this lab is for you to give and receive peer feedback on project outlines before writing up your proposal. 

- **You can find your team's reviewing assignments in the first sheet [here](https://docs.google.com/spreadsheets/d/1_pw_lYkFutMjuL1_j6RdxNyQlj7LvF_f5eEKr1Qm-w0/edit?usp=sharing).**
- **The second sheet contains links to all teams' outlines to review.**
- **Once you have reviewed an outline, please enter a link to your review (e.g. a Google Doc, public github repo, etc.) replacing your team name in the corresponding other team's row in the third sheet.**


Here's a reminder of what your completed project proposal should include:
- Motivation
- Hypotheses (key ideas)
- How you will test those hypotheses: datasets, ablations, and other experiments or analyses.
- Related work and baselines
- I/O: What are the inputs and output modalities? What existing tools will you use to convert device inputs (that are not a core part of your project) to a format readable by the model, and vice versa?
- Hardware, including any peripherals required, and reasoning for why that hardware is needed for this project. (This is where you will request additional hardware and/or peripherals for your project!)
- Will you need to perform training off-device? If so, do you need cloud compute credits (GCP or AWS), and how much?
- Potential challenges, and how you might adjust the project to adapt to those challenges.
- Potential extensions to the project.
- Potential ethical implications of the project.
- Timeline and milestones. (Tip: align with Thursday in-class labs!)

Group name: TripleFighting
---
Group members present in lab today: Jiajun Bao, Songhao Jia, Zongyue Zhao.

1: Review 1
----
Name of team being reviewed: NoName
1. How does your team's background align with that of the proposed project (e.g. if the proposed project is in CV but your team's background is mostly NLP, state that here. Similarly, if your team also has someone who specializes in the area of the proposed project, state that here.)
The project mainly focuses on techniques for on-device ML itself, rather than the application field (CV/NLP). Meanwhile, our group has one person with an NLP background and two people with a CV background. None of us specialized in on-device ML prior to this class.
2. Are the motivation, hypotheses, and plan to test those hypotheses clear? If not, where could the team provide more clarification or explanation?
Most of the explanations are clear. However, I think it would be nice if they can elaborate on how experimenting “on-the-wild” works. Is the data going to be pre-processed to have the same dimension/size as their chosen dataset? Is such pre-processing going to take place on the Jetson board, or on a separate device? Furthermore, will training be included in this process?
3. Does the project seem properly scoped for a semester-long project? Why or why not? How might the team be able to adjust the project to be appropriately scoped?
Yes, the project seems to be properly scoped for a semester-long project. This is because of the structure of this project: multiple deliverables can be worked on in parallel. If they have trouble in making one compression technique work, this issue is not going to be a major roadblock for the entire project.
4. Are there any potential ethical concerns that arise from the proposed project? 
No, I do not think so. The three datasets they are considering are all publicly available, and there is no direct experiment on humans or animals.
5. Any additional comments or concerns? Any related work you know of that might be relevant and/or helpful?
Great work!

2: Review 2
----
Name of team being reviewed: 1 tsp of sugar and 3 eggs
1. How does your team's background align with that of the proposed project (e.g. if the proposed project is in CV but your team's backgorund is mostly NLP, state that here. Similarly, if your team also has someone who specializes in the area of the proposed project, state that here.)
The proposed project is in the field of computer vision and computational graphics. Two members in our team specialized in this area.

2. Are the motivation, hypotheses, and plan to test those hypotheses clear? If not, where could the team provide more clarification or explanation? 
I think most of the parts is clear enough, but some part of them need more explanation.  For instance, why the modsl run on the nano would be expected to have a better performance than the modals run on cell phone?  In the hypothesis section, they mentioned that “Phones' cameras do not perform well under low light conditions.”, and they wanted to “run our baseline inference model with at least one input raw image on jetson nano” to overcome the issue. However, as far as we know, mainstream cell phones nowadays share strong computation power, not to mention the iPad equipped with M1 chip. Also they have better camera than jetson Nano and larger memory. I believe both computation resource and camera limitation would not be the reason to achieve better performance.

3. Does the project seem properly scoped for a semester-long project? Why or why not? How might the team be able to adjust the project to be appropriately scoped?

From our perspective, we think it would be better to add more features than only pruning a model to build and run on nvidia nano, since the workload is somehow lower than a semester-long project. For example, using microphones to help detecting the  human location would be a new feature.

4. Are there any potential ethical concerns that arise from the proposed project? 
I believe not.

5. Any additional comments or concerns? Any related work you know of that might be relevant and/or helpful?
Nope.

3: Review 3
----
Name of team being reviewed: Macrosoft
1. How does your team's background align with that of the proposed project (e.g. if the proposed project is in CV but your team's background is mostly NLP, state that here. Similarly, if your team also has someone who specializes in the area of the proposed project, state that here.) 
The proposed project is in the field of computer vision. Two members in our team specialized in this area.

2. Are the motivation, hypotheses, and plan to test those hypotheses clear? If not, where could the team provide more clarification or explanation? 
Motivation: clear
Hypothesis:
Clear
I/O
You mention that three microphones are used. Could you provide more clarification on how they will be used. If every person is given a microphone and each microphone will record the sound of one person, you will have separate audio for each participant. Then, there is no need for you to have a vision system to recognize who is speaking, since their audio has already been distinguished.   
3. Does the project seem properly scoped for a semester-long project? Why or why not? How might the team be able to adjust the project to be appropriately scoped?
The project is properly scoped for a semester-long project. The authors limit the number of speakers to three, so they don’t have to prepare a high-performance vision module. Also, there are already pre-trained recognition models for them to get started with. They can spend more time on the deployment and power tuning.
4. Are there any potential ethical concerns that arise from the proposed project? 
No, the authors are using data collected on themselves. There are no ethical issues as long as they use their product internally. 
5. Any additional comments or concerns? Any related work you know of that might be relevant and/or helpful?
The project is interesting and challenging. Nice jobs!
4: Receiving feedback
----
Read the feedback that the other groups gave on your proposal, and discuss as a group how you will integrate that into your proposal. List 3-5 useful pieces of feedback from your reviews that you will integrate into your proposal:
1. Privacy concerns: all our reviewers proposed concerns regarding collecting data from random CMU students without their consent. This is a valid argument. Thus, we will not gather facial images on the fly, instead, we will use publicly available mask datasets and augment them with vocal time series.
2. Some reviewers believe that it would be better to show the details about how we train and infer our dataset, especially where we train the model and how to use the microphone and camera. We would include more details for better illustration. For now,  we would train the model on a cloud server, and inference it using our device. During inference, only one of the microphone and camera would be working: microphone would be used to detect if someone is coming, and when it finds someone, the microphone would stop working and camera starts to detect mask; after a certain amount of time the camera could not find anyone, it would stop and pass the work to the microphone. 
3. One of our reviewers suggested investigating the location of the mask - whether the mask is properly worn over the nose and mouth. This makes sense, and at this stage we are considering using this [dataset](https://arxiv.org/abs/2008.08016), which distinguishes the cases whether a mask is present but not properly worn and the correct usage.

You may also use this time to get additional feedback from instructors and/or start working on your proposal.
