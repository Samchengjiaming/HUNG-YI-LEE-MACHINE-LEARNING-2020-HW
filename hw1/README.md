# ***First assignment***
### <b>Description of the problem</b>
[the link of assignment](https://docs.google.com/presentation/d/18MG1wSTTx8AentGnMfIRUp8ipo8bLpgAj16bJoqW-b0/edit#slide=id.g4cd6560e29_0_10) 
In the context of the problem,the content of 18 substances in the atmosphere will affect the concentration of PM2.5.given,the content of these 18 substances in the atmosphere in the first eight hours,we need manual achieve linear regression to forecast the PM2.5 of next hour.<br>
### <b>Data preprocessing</b>
- Due to the problem of the data character encoding set, we changed all Chinese in the data to English, but this did not affect the results of our model training.
- The'NR' in the data indicates that there was NO RAIN on this day. Considering that the features input to the model are all floating-point data,so we transform 'NR' to zero. 
### <b>Model</b>
we assume the model set as follow:  
model 1.***y=b+W<sub>1</sub>X.***<br/>
model 2.***y=b+W<sub>1</sub>X+W<sub>2</sub>X<sup>2</sup>.***<br/>
model 3.***y=b+W<sub>1</sub>X+W<sub>2</sub>X<sup>2</sup>+W<sub>3</sub>X<sup>3</sup>.***<br/>
the Y represents the predicted value of PM2.5,X is the input feature,X<sub>i</sub><sup>j</sup>indicates the content of the i-th substance in the atmosphere at the t-th time.

### <b>Result</b>
#### <b>the model 1:<b>
***y=b+W<sub>1</sub>X.***　　[code>>](./code/linear%20model_1.py)  
when the learning_rate=100,epoch=1000,loss during training：  
![error_img_01](img/linearModel_1.png)  
#### <b>the model 2:<b>
***y=b+W<sub>1</sub>X+W<sub>2</sub>X<sup>2</sup>.***　　[code>>](./code/linear%20model_2.py)  
when the learning_rate=100,epoch=1000,loss during training：  
![error_img_01](img/linearModel_2.png)  
#### <b>the model 2:<b>
***y=b+W<sub>1</sub>X+W<sub>2</sub>X<sup>2</sup>+W<sub>3</sub>X<sup>3</sup>.***　　[code>>](./code/linear%20model_3.py)  
when the learning_rate=100,epoch=1000,loss during training：  
![error_img_01](img/linearModel_3.png)  
Under the same learning rate and training times, the loss value of model one about 8.5, 
the loss value of model two about 10, and the loss value of model three about 11.2.
This situation is not in line with common sense. Generally speaking, 
the more complex the model, the better the performance on the training set.  
who knows?
