# Classifying Heatmaps with CNN

This project solves classification problem by training CNNs on heatmaps and scanpathes from an eye tracking study. 
In the paper of original study, logistic regression was chosen to solve the problem with numerical data. The eye tracker software also produces heatmaps and scanpathes of eye gaze, this information was used to solve the classification problem. 
While this is an unusual approach for such problems, it has some benefits in comparison to numerical method. 
First of all, because of the heatmaps we don't have to choose between saccades and fixations as in the numerical method. These two features are strongly correlated which suggests only using one of them for machine learning approaches. 
Second of all, this approach has promising in terms of interpretability from visual perspective. While neural networks are clearly harder to interpret than simpler models, visualizing the gradients on heatmaps can be a very good tool for interpretability.
Finally, 
