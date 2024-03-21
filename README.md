# Classifying Heatmaps with CNN

This project solves classification problem by training CNNs on heatmaps and scanpathes from an eye tracking study (https://dl.acm.org/doi/10.1145/3588015.3590118). 

In the paper of original study, logistic regression was chosen to solve the problem on numerical data. The eye tracker software also produces heatmaps and scanpathes of eye gaze, this information was used in this project.

While this is an unusual approach for such problems, it has some benefits in comparison to numerical method. 

First of all, because of the heatmaps we don't have to choose between saccades and fixations as in the numerical method. These two features are strongly correlated which suggests only using one of them for machine learning approaches. 

Second of all, this approach is promising in terms of interpretability from visual perspective. While neural networks are clearly less interpretable than simpler models, visualizing the gradients on images can be a very good tool for interpretability.

## Architecture 

## Future work



