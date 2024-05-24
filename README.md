# deep-learning-challenge
Module 21 challenge

## Overview
The purpose of this analysis was to create a model that would help Alphabet Soup select/predict potential successful applicants. Using a dataset with more than 34,000 companies that Alphabet Soup has previously funded, I implemented machine learning and neural network techniques to create a model that aimed to achieve at least 75% accuracy in selecting successful ventures.

-----------------------------------

## Results

### Data Preprocessing
* Model Target (Dependent Variable): the column "IS_SUCCESSFUL" was chosen as my target, since a successful company was the ultimate goal of each applicant. Each applicant's success was marked with a 0 or a 1.
* Model Features (Independent Variables): after preprocessing the data, the chosen features were APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, and ASK_AMT for the original model (before optimization). In the optimization portion, I kept all features except SPECIAL_CONSIDERATIONS. Figure 1 shows the original preprocessed data.
  
![alt text](ScreenShots/02_orginal_dropped_cols.png)
  
  Fig 1, original preprocessed data

* Removed Variables (no impact on features or target): in the original model, only the EIN and NAME columns were dropped. In the optimization file I also dropped SPECIAL_CONSIDERATIONS. I chose to remove that feature due to the majority of the values being N and a very small percentage being special considerations (~0.08%), and I did not think that this would have much, if any, impact on the target. See figure 2 for the optimized preprocessed data.

![alt text](ScreenShots/03_opti_dropped_cols.png)
  
  Fig 2, optimized preprocessed data without SPECIAL_CONSIDERATIONS

### Compiling, Training, and Evaluating the Model
#### How many neurons, layers, and activation functions did you select for your neural network model, and why?
#### Original Model:
* I started with 2 hidden layers and 3 nodes in each hidden layer. The loss was significant at around 8, and the accuracy was around 0.42. I used 100 epochs in this iteration
* Using Xpert Learning Assistant to help determine the number of nodes (it recommended using a number between the number of inputs and outputs), I chose 10 to see how the loss and accuracy were affected. I chose to keep the activation for each layer and the output layer to be relu since it is usually the default. This resulted in with a loss of 0.597 and an accuracy of 0.725. Figures 3 and 4 show the original layers, nodes, output, and evaluation scores.

![alt text](ScreenShots/04_original_model_layers.png)

![alt text](ScreenShots/05_original_model_eval.png)
  
  Figs 3 and 4, original model layers and evaluation

#### What steps did you take in your attempts to increase model performance?
#### Optimization Attempt # 1:
* I chose to keep 2 hidden layers and 10 nodes in each layer. The hidden layers still used a relu activation, but I changed the output layer to a sigmoid since the output range was either 0 or 1. The only other change I made was increasing the number of epochs to 150.
* This attempt yielded a loss of 0.553 and an accuracy of 0.726, which is slightly better than the original model. Figures 5 and 6 show the model and evaluation of optimization attempt # 1.

  ![alt text](ScreenShots/06_opti_attempt_1_structure.png)

  ![alt text](ScreenShots/07_opti_attempt_1_eval.png)

Figs 5 and 6: Optimization attempt # 1 model and evaluation.
  
#### Optimization Attempt # 2:
* This attempt I still kept 2 hidden layers and 10 nodes in the first layer, but I decreased to 8 nodes in the second layer. I chose to keep the activations the same in each layer as attempt # 1, as well as 150 epochs.
* I decided to keep the model similar since the original model and attempt # 1 both had an accuracy of just under 0.750, and I did not want to change too many things that may have affected the accuracy drastically.
* This attempt yielded a loss of 0.554 and an accuracy of 0.726, nearly exactly the same as attempt # 1. Figures 7 and 8 show the model and evaluation of optimization attempt # 2.

  ![alt text](ScreenShots/08_opti_attempt_2_structure.png)

  ![alt text](ScreenShots/09_opti_attempt_2_eval.png)

Figs 7 and 8: Optimization attempt # 2 model and evaluation.
  
#### Optimization Attempt # 3:
* My final attempt at optimization was to employ the auto-optimizer kerastuner. Using module 21-2 activities 04 and 05 as guidelines, I set the tuner to create a sequential model with hyperparameter options. I allowed kerastuner to choose between relu, tanh, and sigmoid activations. In my first iteration, I set the tuner to choose between 1 and 41 neurons in the first layer (41 being the limit because there is a total of 41 columns in the dataframe used); 1 to 5 hidden layers with 1 to 15 neurons each; set the output activation as sigmoid; and set the tuner to run a maximum of 100 epochs with 5 iterations. This set up proved to be inefficient and allowed too many variations to be made, as it took over 6 hours to run and the best accuracy it could get was 0.7299. See figure 9.

![alt text](ScreenShots/10_auto_opti_attempt1.png)

Fig 9: auto-optimization attempt # 1 accuracy and run time

* After consulting with classmates and my instructional team, I tweaked the tuner to only choose a maximum of 20 neurons in the first hidden layer, with a step of 4; a maximum of 10 neurons in the second hidden layer with a step of 3; and set the tuner to run a maximum of 20 epochs and 2 iterations. This model only took 24 minutes to run, and the accuracy was only marginally better with the best being 0.7301. Figures 10 and 11 show the code used (which includes some code from the first auto-attempt commented out), and figure 12 shows the run time and accuracy score.

  ![alt text](ScreenShots/10_auto_opti_attempt.png)
  ![alt text](ScreenShots/11_auto_opti_final.png)

  Figs 10 and 11: code for the second auto-optimization attempt


  ![alt text](ScreenShots/12_auto.png)

  Fig 12: best accuracy score and run time for the second auto-optimization attempt

**Were you able to achieve the target model performance?**
Unfortunately I was not able to achieve the target model performance of 0.750 (75%). The closest that I was able to get was 0.7301 (73.01%).


-----------------------------------
## Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.
Overall, the closest that I was able to get to 0.750 accuracy was 0.7301. I do not think that I changed the model settings enough (in either the manual optimizations or the auto-optimizations) to create a drastic difference in accuracy. 

My recommendations for other models to try would be either a random forest or a decision tree. I believe with the binning/separating out columns, random forest might be able to more efficiently tackle the different predictions. The decision tree might be a better fit though, because it shows each decision process at each level.
