## CS4641 Group 15 Fall 2020

### Project: Using Machine Learning to Aid in the Mitigation of Wildfires

### Project Graphic/Overview:  


![Project Visual](project_outline.jpg)  


### Summary/Overview:

Our dataset that we are using contains information about 1.88 million US Wildfires. For each sample, there are data points on the location, name of the fire, date and time of the fire, size of the fire, as well as several codes corresponding to different fire classes that each fire is classified as. We are hoping to take this data and create meaningful insights that can aid in the mitigation efforts of wildfires, for example predicting the final size of a fire, predicting the cause of a fire, or predicting where and when fires are most likely to start. We plan to start by using unsupervised learning to focus on the clustering of different types of fires in order to get a better since of the data we are dealing with in terms of causes, sizes, and locations, and then we can move to supervised learning which will allow us to use tools like regression in order to predict the ultimate size of a fire given certain features of a fire that has started. When dealing with this regression step, we also anticipate pulling in other weather data that can help us build a succesful regression model. This is important to be able to predict as it has the most direct effect on people's lives - more so than cause or location of a fire. People need to know how big a fire is going to get and if we can build a tool that can accurately give them this information it could play an important role in keeping people safe.  

### Background:

There have previously been attempts by data scientists to deal with wildfires using machine learning. For example in reference (2), scientists attempted to predict the size of a wildfire given that it already started. They had a slightly more specific goal, which was analyzing the different types of kindling and how that affects burn area in Alaska. They were successfully able to identify certain natural features that would lead to fires burning out of control in that region. Another study focused on estimating population exposure to fires, which is interesting because it could potentially be very useful in terms of evacuation efforts and saving lives.

### Methods:

#### Unsupervised Section:  
##### Data Cleaning:  
Getting rid of features:  

Feature analysis:  

##### Unsupervised Algorithms:

During this unsupervised learning phase, we attempted to employ several clustering and density estimation algorithms to get a better understanding of our data. The first feature we looked at was location, as we thought that clustering by location might prove meaningful to split the data up. However, we ended up with a plot that didn't tell us much information, as it turned out to simply be a map of the United States (indicating that there are fires everywhere- as expected). See the plot below:  
![Elbow Method](/vis/LocationElbow.jpg)  
![Clustering on Location](/vis/LocationCluster.jpg)  
Since this plot tells us nothing besides what we expected to be true, and our goal is to predict the size of a fire, we wanted to get a better estimator of how the sizes of fires were distributed on a map. For this we decided to use a Kernel Density Estimator, as it would tell us not only where fires occur but also how frequently (dense) fires occur in a certain location. To fully understand our data we ran this Kernel Density Estimator for each class of fire (A-G), and received the following results:  

Class A (smallest):  
![Class A Fires](/vis/ClassAHM.jpg)  
Class B:  
![Class B Fires](/vis/ClassBHM.jpg)  
Class C:  
![Class C Fires](/vis/ClassCHM.jpg)  
Class D:  
![Class D Fires](/vis/ClassDHM.jpg)  
Class E:  
![Class E Fires](/vis/ClassEHM.jpg)  
Class F:  
![Class F Fires](/vis/ClassFHM.jpg)  
Class G:  
![Class G Fires](/vis/ClassGHM.jpg)  








#### Supervised Section:  
Finally, after now having gained a thorough understanding of the data, we hope to be able to predict the size of a wildfire given certain characteristics of it. In this step we anticipate having to pull in data from a weather API in order to gain more features for each data point. Some of the methods we plan on using in this step include:  
  1. Data cleaning/engineering with wildfires and weather  
  2. Supervised learning models including, but not limited to:  
      * Linear Regression
      * Ridge Regression


### Results:

The hypothetical outcome for this project is that we will be able to predict the final size of a fire given that the fire started, as well as predict the causes of fires given that they started. Both of these things would be useful in terms of fire prevention, as they would help the experts better understand the behavior of fires and possibly what steps need to be done in order to slow their spread or prevent them from happening.

### Discussion:

For this project, the best outcome would be that we are able to successfully predict certain aspects of a wildfire like size, cause, and area. The worst possible outcome to our experiment would be a lack of correlation or success with our various models, which would mean that our models cannot accurately predict features of a wildfire, either because the fires themselves are random in nature or because our models were not built correctly. Regardless of the outcome we receive, there are clear next steps that can be taken. For example, if our model is unsuccessful, some of the next steps might be to look at the dataset and see why the fires appear to act so randomly, and then try and take into account this randomness with a future model if possible. If we are successful in our wildfire prediction efforts, a logical next step would be to try and scale our model to other datasets (outside of the United States, for example Italy) where they are also having trouble with wildfires, and see how a similar model would perform in a different general geographical region.

### References:
(1): [A Machine Learning-Based Approach for Wildlife Susceptibility Mapping. The Case Study of the Liguria Region in Italy](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwidsPeivf_rAhVSTd8KHd8yBj8QFjAEegQICRAB&url=https%3A%2F%2Fwww.mdpi.com%2F2076-3263%2F10%2F3%2F105%2Fpdf&usg=AOvVaw1YcrKdNxaP00bDtDOdOn87)

(2): [Machine Learning used to help tell which wildfires will burn out of control](https://www.sciencedaily.com/releases/2019/09/190917133052.htm)

(3): [Spatiotemporal Prediction of Fine Particulate Matter During the 2008 Northern California Wildfires Using Machine Learning](https://www.firescience.gov/projects/14-1-04-5/project/14-1-04-5_EnvSciTech_Reid_SpatiotemporalModelingWildfires.pdf)

(4): [Global trends in wildfire and its impacts: perceptions versus realities in a changing world](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4874420/)  

(5): [Distribution of Lightning- and Man-Caused Wildfires in California](https://www.fs.fed.us/psw/publications/documents/psw_gtr058/psw_gtr058_6a_keeley.pdf)  

(6): [Observed Impacts of Anthropogenic Climate Change on Wildfire in California](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019EF001210)  
