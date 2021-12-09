# Regression

Regression is a [Supervised Learning](https://en.wikipedia.org/wiki/Supervised_learning#:~:text=Supervised%20learning%20(SL)%20is%20the,on%20example%20input%2Doutput%20pairs.&text=A%20supervised%20learning%20algorithm%20analyzes,used%20for%20mapping%20new%20examples.) technique which helps in finding the correlation between variables and enables us to predict the continuous output variable based on the one or more predictor variables.

In Regression, we plot a graph between the variables which best fits the given datapoints, using this plot, the machine learning model can make predictions about the data. In simple words, "***Regression shows a line or curve that passes through all the datapoints on target-predictor graph in such a way that the vertical distance between the datapoints and the regression line is minimum.***" The distance between datapoints and line tells whether a model has captured a strong relationship or not.

## Simple Linear Regression

Simple Linear Regression is a type of Regression algorithms that models the relationship between a dependent variable and a single independent variable. The relationship shown by a Simple Linear Regression model is linear or a sloped straight line, hence it is called Simple Linear Regression.

The key point in Simple Linear Regression is that the ***dependent variable must be a continuous/real value***. However, the independent variable can be measured on continuous or categorical values.

The Simple Linear Regression model can be represented using the below equation:

```
y = b0 + b1*x 
```

Where,

**y = It is the Dependent variable**

**x = It is the Independent variable**

**b0 = It is the intercept of the Regression line (can be obtained putting x=0)**

**b1 = It is the slope of the regression line, which tells whether the line is increasing or decreasing.**

![linear regression](https://i.ibb.co/vxw8LsZ/Screenshot-27.png)
