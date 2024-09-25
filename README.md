# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4.Define a function for costFunction,cost and gradient.
5.Define a function to plot the decision boundary
6.Define a function to predict the Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: VIGNESH M
RegisterNumber: 212223240176
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```

## Output:
## ARRAY VALUE OF X
![322345559-adba9bb9-8ee7-4521-865f-6541556e98ef](https://github.com/rajalakshmi8248/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122860827/03ed711c-7af2-44fb-9215-a19800962b31)
## ARRAY VALUE OF Y
![322345635-c139e7cf-d762-4166-86bd-41e30276be31](https://github.com/rajalakshmi8248/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122860827/a1ea5288-4629-4ab2-bac4-fdb57567c2bb)
## EXAM1:SCORE GRAPH
![322345774-a8bb8788-e304-4b17-a64f-5e419af27a43](https://github.com/rajalakshmi8248/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122860827/b9d7803d-e87a-474c-94e0-d50f507dff2b)
## SIGMOID FUNCTION GRAPH
![322345917-7e30dc25-1f43-4009-adda-b125bdc7bf43](https://github.com/rajalakshmi8248/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122860827/3b833922-2ab6-4e0a-aed9-ac530e860844)
## X_train_grad value
![322346007-0488c1ba-5dd5-4051-a740-a93a1c75e795](https://github.com/rajalakshmi8248/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122860827/6b94152e-2104-4eeb-b50b-045bb0ce3f31)
## Y_train_grad value
![322346048-45d08099-e177-47d2-941a-3654cfbb76bf](https://github.com/rajalakshmi8248/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122860827/2880472d-f386-4932-a4eb-46a4f4cf807d)
## Print resx
![322346154-de9a4fb6-4f29-4683-83cd-44c8f373090d](https://github.com/rajalakshmi8248/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122860827/9ccda9d4-fda4-4f92-9fd6-97372dfd80d7)
## Decision boundary - graph for exam score
![322346232-746c6c23-aced-4649-aeaf-1b217de89468](https://github.com/rajalakshmi8248/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122860827/eba20280-d813-4a66-aa88-b6dcc1743b7c)
## Probability value
![322346327-643c5118-4c95-4595-a72f-eb7eef062130](https://github.com/rajalakshmi8248/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122860827/62b1a460-68fd-447b-9e65-313899c06e0e)
## prediction value of mean
![322346441-b2724af8-10df-4290-b81f-392cc44f71d2](https://github.com/rajalakshmi8248/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/122860827/aa7d64c0-9405-4330-a656-83fc0ea832e4)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
