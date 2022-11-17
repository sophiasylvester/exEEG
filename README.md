# exEEG

In this project we explore the use of deep learning and explainable AI for EEG data. 

## Data
The first dataset we use stems from a psychological experiment with two conditions. The data cannot be published or details disclosed because of data privacy concerns. 

## Multi-task classification
We trained a multi-task CNN which firstly predicts which trials stem from which participant, and secondly which condition was realised in that trial. Different weightings of the task are compared in order to find the best configuration. The idea behind the multi-task architecture was that EEG data from different persons varies substantially, hence the prediction of the participant could help boost the performance of the second task.




