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


![simple linear regression](https://i.ibb.co/vxw8LsZ/Screenshot-27.png)


## Multiple Linear Regression

Multiple Linear Regression is an extension of Simple Linear regression as it takes more than one predictor variable to predict the response variable. We can define it as:

***Multiple Linear Regression is one of the important regression algorithms which models the linear relationship between a single dependent continuous variable and more than one independent variable.***

```
y = b0 + b1*x1 + b2*x2 + b3*x3 + ... (+ b4*d1 + b5*d2 + ...)
```

Where,

**y = Output/Response/Dependent variable**

**b0, b1, b2, b3, ... bn = Coefficients of the model**

**x1, x2, x3, x4, ... xn = Various Independent/feature variable**

**d1, d2, d3, ... bn-1 = Dummy variables**

How is the coefficient b0 related to the dummy variable trap?
Since D2 = 1 − D1 then if you include both D1 and D2 you get:

                          y = b0 + b1x1 + b2x2 + b3x3 + b4D1 + b5D2
                            = b0 + b1x1 + b2x2 + b3x3 + b4D1 + b5(1 − D1)
                            = b0 + b5 + b1x1 + b2x2 + b3x3 + (b4 − b5)D1
                            = *b0 + b1x1 + b2x2 + b3x3 + *b4D1
                            
with *b0 = b0 + b5 and *b4 = b4 − b5

Therefore the information of the redundant dummy variable D2 is going into the constant b0.


>Note - If there are N dummy variable table, omit one and consider only N-1 tables

![multiple linear regression](https://i.imgur.com/GHAtNx9.png)


## Polynomial Regression

Polynomial Regression is a regression algorithm that models the relationship between a dependent(y) and independent variable(x) as nth degree polynomial. The Polynomial Regression equation is given below:

![multiple linear regression](https://i.imgur.com/H9vVcQf.png)

Hence, ***"In Polynomial regression, the original features are converted into Polynomial features of required degree (2,3,..,n) and then modeled using a linear model."***

> Note: A Polynomial Regression algorithm is also called Polynomial Linear Regression because it does not depend on the variables, instead, it depends on the coefficients, which are arranged in a linear fashion.

![poly](https://i.ytimg.com/vi/2wzxzHoW-sg/maxresdefault.jpg)


