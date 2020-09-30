## CS4641 Group 15 Fall 2020

### Project: Using Machine Learning to Aid in the Mitigation of Wildfires

### Graphic:  





### Summary/Overview:

Our dataset that we are using contains information about 1.88 million US Wildfires. For each sample, there are data points on the location, name of the fire, date and time of the fire, size of the fire, as well as several codes corresponding to different fire classes that each fire is classified as. We are hoping to take this data and create meaningful insights that can aid in the mitigation efforts of wildfires, for example predicting the final size of a fire, predicting the cause of a fire, or predicting where and when fires are most likely to start.

### Background:

There have previously been attempts by data scientists to deal with wildfires using machine learning. For example in reference (2), scientists attempted to predict the size of a wildfire given that it already started. They had a slightly more specific goal, which was analyzing the different types of kindling and how that affects burn area in Alaska. They were successfully able to identify certain natural features that would lead to fires burning out of control in that region. Another study focused on estimating population exposure to fires, which is interesting because it could potentially be very useful in terms of evacuation efforts and saving lives. 

### Methods:

For initial data analysis:  
-Correlation Matrix  
-Regression models?  
-Encoding of non-numeric values  
Machine Learning methods:  
-Gaussian Mixture Models to identify likely areas that wildfires may start  
-K-Means clustering to group certain sizes of wildfires  
-Heirarchial clustering to narrow down certain types of causes of wildfires  


### Results:

The hypothetical outcome for this project is that we will be able to predict the final size of a fire given that the fire started, as well as predict the causes of fires given that they started. Both of these things would be useful in terms of fire prevention, as they would help the experts better understand the behavior of fires and possibly what steps need to be done in order to slow their spread or prevent them from happening.

### Discussion:

For this project, the best outcome would be that we are able to successfully predict certain aspects of a wildfire like size, cause, and area. The worst possible outcome to our experiment would be a lack of correlation or success between our training and testing data, which would mean that our models cannot accurately predict features of a wildfire, either because the fires themselves are random in nature or because our models were not built correctly. Regardless of the outcome we receive, there are clear next steps that can be taken. For example, if our model is unsuccessful, some of the next steps might be to look at the dataset and see why the fires appear to act so randomly, and then try and take into account this randomness with a future model if possible. If we are successful in our wildfire prediction efforts, a logical next step would be to try and scale our model to other datasets (outside of the United States, for example Italy) where they are also having trouble with wildfires, and see how a similar model would perform in a different general geographical region.

### References:

(1):  
https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwidsPeivf_rAhVSTd8KHd8yBj8QFjAEegQICRAB&url=https%3A%2F%2Fwww.mdpi.com%2F2076-3263%2F10%2F3%2F105%2Fpdf&usg=AOvVaw1YcrKdNxaP00bDtDOdOn87

(2):  
https://www.sciencedaily.com/releases/2019/09/190917133052.htm

(3):  
https://www.firescience.gov/projects/14-1-04-5/project/14-1-04-5_EnvSciTech_Reid_SpatiotemporalModelingWildfires.pdf 

