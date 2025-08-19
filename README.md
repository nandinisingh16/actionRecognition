#  Human Action Recognition using CNN + LSTM (TensorFlow)

Convolutional Neural Networks (CNNs) are excellent for **image data**, while Long Short-Term Memory (LSTM) networks excel at **sequential data**. When combined, they provide the **best of both worlds** — allowing us to solve challenging **video classification problems**.

In this tutorial-style notebook, we implement **Human Action Recognition** using **CNN + LSTM architectures** in TensorFlow.  
We experiment with two different approaches:
- **ConvLSTM**  
- **LRCN (Long-term Recurrent Convolutional Network)**  

Finally, we evaluate the best model on **YouTube videos**.

---

## Table of Contents
1. [Overview](#overview)  
2. [Dataset](#dataset)  
3. [Approaches](#approaches)  
4. [Implementation Steps](#implementation-steps)  
5. [Results](#results)  
6. [Future Work](#future-work)  
7. [Author](#author)  

---

##  Overview
- **Image Classification** → Single frame prediction. Ignores temporal sequence.  
- **Video Classification** → Must capture both **spatial features** (appearance) and **temporal features** (motion).  

 A **video = sequence of frames**. Simply applying image classifiers frame-by-frame can lead to errors, as temporal relationships are ignored. Example: a "Backflip" video may look like "Falling" if analyzed frame by frame.  

To overcome this, we explore several strategies:
- Single-frame classification  
- Late fusion & early fusion  
- 3D CNNs  
- Pose detection + LSTMs  
- **CNN + LSTM (our chosen method)** 

---

##  Dataset
We use:  
- **UCF50**: [UCF50 Dataset](https://www.crcv.ucf.edu/data/UCF50.php)  
- (Optional extension) **UCF101**: [UCF101 Dataset](https://www.crcv.ucf.edu/data/UCF101.php)  

Each dataset consists of labeled human action videos (e.g., Basketball, Swing, TaiChi, WalkingWithDog).  

---

##  Approaches Implemented
We implemented **two CNN+LSTM architectures**:

1. **ConvLSTM**  
   - Uses convolutional operations inside the LSTM cells.  
   - Captures both **spatial** (CNN) and **temporal** (LSTM) features jointly.  

2. **LRCN (Long-term Recurrent Convolutional Network)**  
   - Extracts frame-level features using CNN.  
   - Feeds them into LSTM layers for sequential modeling.  

---

## 🛠️ Implementation Steps
1. **Download & Visualize Data**  
   - Load dataset, inspect classes, preview sample frames.  

2. **Preprocess Dataset**  
   - Extract frames from videos.  
   - Resize to 64×64.  
   - Select 20 frames per video.  
   - Normalize pixel values.  

3. **Train/Test Split**  
   - Create balanced training and testing sets.  

4. **ConvLSTM Approach**  
   - Build ConvLSTM model.  
   - Compile & train.  
   - Plot accuracy and loss curves.  

5. **LRCN Approach**  
   - Build LRCN model (CNN feature extractor + LSTM).  
   - Compile & train.  
   - Plot accuracy and loss curves.  

6. **Evaluation on YouTube Videos**  
   - Load external videos via `pafy` + `youtube-dl`.  
   - Preprocess and predict action class.  

---

## Results
- ConvLSTM achieved ~**78% accuracy** on test set (with 4 sample classes).  
- Models were able to predict real-world YouTube videos with reasonable confidence.  
- Accuracy improves with:  
  - More training epochs  
  - Larger number of classes  
  - Pretrained feature extractors (e.g., MobileNetV2, ResNet)  

---

## Future Work
- Train on **full UCF101** dataset.  
- Use advanced architectures: **3D CNNs, Transformers for video**.  
- Deploy model as a **web app** for real-time video action recognition.  

---
## Output
<img width="930" height="225" alt="image" src="https://github.com/user-attachments/assets/6ec0da25-0cb2-47fc-863b-ad8bda11c645" />
<img width="472" height="275" alt="image" src="https://github.com/user-attachments/assets/200b4276-5475-4ab1-9af2-1948b7a7f9cc" />
<img width="467" height="271" alt="image" src="https://github.com/user-attachments/assets/f16ef77a-b0e3-4d97-b550-9a5bb55d8be4" />


