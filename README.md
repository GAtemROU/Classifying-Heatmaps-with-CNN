# Classifying Heatmaps with CNN

This project solves classification problem by training CNNs on heatmaps and scanpathes from an eye tracking study (https://dl.acm.org/doi/10.1145/3588015.3590118). Here is an example of the input images:

<img src="https://github.com/GAtemROU/Classifying-Heatmaps-with-CNN/assets/105051372/6433069d-b4c7-428d-aa33-73f3c0cc83ae" height='100' width='650'> <img src="https://github.com/GAtemROU/Classifying-Heatmaps-with-CNN/assets/105051372/8e51b992-1fb6-494f-9bca-b777458b69f2" height='100' width='650'>

 The study investigates into how ones eye gaze behaves during reading of unknown words.

In the paper of original study, logistic regression was chosen to solve the problem on numerical data. The eye tracker software also produces heatmaps and scanpathes of eye gaze, this information was used in this project.





While this is an unusual approach for such problems, it has some benefits in comparison to numerical method. 

First of all, because of the heatmaps we don't have to choose between saccades and fixations as in the numerical method. These two features are strongly correlated which suggests only using one of them for machine learning approaches. 

Second of all, this approach is promising in terms of interpretability from visual perspective. While neural networks are clearly less interpretable than simpler models, visualizing the gradients on images can be a very good tool for interpretability.

## Architecture 
The architecture was kept rather simple. 
Firstly, batchnormalization without learnable parameters was applied to the input.

Then, there are two convolutional layers followed by one fully connected layer. 

Both convolutional layers look as follows:
Conv->ReLU->Max Pool->Dropout

The convolitions have sizes and strides of 3, 1 and 5, 2 respectively. 


## Evaluation
To evaluate models better, cross validation was implemented. After training, the model was tested on 4 unseen participants. Then, the model was trained from scratch with a different set of test participants. The process is repeated untill each participant have been to the test set ones. 

## Performance
The evaluation mentioned above results in the average f1 score of **0.89**, which that the CNN reaches about the same performance as the logisitc regression in the original study.

## Visualization of gradient
Here is an example of how we can interpret the network by means of gradient.

Here the sentence did not have any unknown word and the prediction of network was correct. We see that the gradient is distributed among the whole sentence, however, the the saturation increases in the middle. Which means that the network was mostly expecting the unkown word to appear in the middle of the sentence.

<img src="https://github.com/GAtemROU/Classifying-Heatmaps-with-CNN/assets/105051372/0f955e04-ac4f-4cf8-88df-76eaadd74190" height='400' width='400'><img src="https://github.com/GAtemROU/Classifying-Heatmaps-with-CNN/assets/105051372/4ee8b179-0dab-4ce3-9cd8-7623eccb1025" height='400' width='400'>

Here the sentence has an unknown word close to the end, the network wrongly predicted the absence of an unknown word. Again we see even more saturated colors in the middle. This shows that model considered middle of the sentence which does not have an unknown word and we see a very steady pattern of saccades and fixations. 

<img src="https://github.com/GAtemROU/Classifying-Heatmaps-with-CNN/assets/105051372/475b2e5e-01f6-461e-8df9-c4e97feacf5c" height='500' width='500'><img src="https://github.com/GAtemROU/Classifying-Heatmaps-with-CNN/assets/105051372/15a15a82-80fe-4d89-9cda-946282083833" height='500' width='500'>













