### ***This Repository stores the homework answer of HungYi Lee MACHING LEARNING(2020).***

I intend to use Python & PyTorch to deal the homework,of course in the first two assignments,I will use numpy replace PyTorch.And I will explain my own thinking in the README for every assignment.

# ***First assignment***
## <b>Description of the problem</b>
[the link of assignment](https://docs.google.com/presentation/d/18MG1wSTTx8AentGnMfIRUp8ipo8bLpgAj16bJoqW-b0/edit#slide=id.g4cd6560e29_0_10) 
In the context of the problem,the content of 18 substances in the atmosphere will affect the concentration of PM2.5.given,the content of these 18 substances in the atmosphere in the first eight hours,we need manual achieve linear regression to forecast the PM2.5 of next hour.<br>
## <b>model</b>
we assume the function set as follow:<br/>
1.***y=b+W<sub>1</sub>X.***<br/>
2.***y=b+W<sub>1</sub>X+W<sub>2</sub>X<sup>2</sup>.***<br/>
3.***y=b+W<sub>1</sub>X+W<sub>2</sub>X<sup>2</sup>+W<sub>3</sub>X<sup>3</sup>.***<br/>
the Y represents the predicted value of PM2.5,X is the input feature,X<sub>i</sub><sup>j</sup>indicates the content of the i-th substance in the atmosphere at the t-th time.





