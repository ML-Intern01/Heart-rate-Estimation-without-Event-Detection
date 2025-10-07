**Heart Rate Estimation without Event Detection **

Problem Statement : Develop a robust and efficient method for continuously estimating heart rate from physiological signals without relying on explicit beat detection or segmentation.

Technical Document Link : 
                        
                        https://docs.google.com/document/d/1392ogqvK8laxwUWLTo2hF6wrp0b69csqyO0vlmgv_E8/edit?usp=drive_link

Data : Google drive link  
      There are 2 folders within the link 
            1) H_R data : Combined all the Heart Sound audio files from the different sources
            2) Different Sources : This folder contatins the data which was collected from the different sources (AiSteth, R-Peak, Open-website, ZCH datasets).

                     https://drive.google.com/drive/folders/10EMZPiZ-aOcCc4bqbWYVQ-oaczaCIK_J?usp=drive_link
              

Preprocessing : 
                1) Butterworth Bandpass filter
                2) Resampled to 1000 Hz
                3) Z-Normalization 
File Name :  Preprocessing.py

Feature Extraction : Extracted Mel-Spectrogram as feature and saving this features as .npy file 
File Name : Feature Extraction.py

Model Training :
                Model Used  - BiLSTM
                Layers that are used for the experiments 
                    - Simple BiLSTM 
                    - Global Max Pooling
                    - Global Average Pooling 
                    - Mean across features
                    - Mean across time steps 
                    - Flatten 



