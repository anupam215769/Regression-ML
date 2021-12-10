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

**How to find best degree?** 

The main form of finding a good fit is to plot the model and see what it looks like visually. You simply test several degrees and you see which one gives you the best fit. The other option is to find the lowest root-mean-square error (RMSE) for your model, but in that case be careful not to overfit the data.

> Note: A Polynomial Regression algorithm is also called Polynomial Linear Regression because it does not depend on the variables, instead, it depends on the coefficients, which are arranged in a linear fashion.

#### Feature Scaling is not needed because, since y is a linear combination of x and x 2 , the coefficients can adapt their scale to put everything on the same scale. For example if y takes values between 0 and 1, x takes values between 1 and 10 and x 2 takes values between 1 and 100, then b1 can be multiplied by 0.1 and b2 can be multiplied by 0.01 so that y, b1x1 and b2x2 are all on the same scale

![poly](https://i.ytimg.com/vi/2wzxzHoW-sg/maxresdefault.jpg)


## Support Vector Regression (SVR)

Support Vector Regression is a supervised learning algorithm that is used to predict discrete values. Support Vector Regression uses the same principle as the SVMs. 

The basic idea behind SVR is to find the best fit line. Support Vector Regression is a supervised learning algorithm that is used to predict discrete values. Support Vector Regression uses the same principle as the SVMs. The basic idea behind SVR is to find the best fit line.

**when to use SVR?**

You should use SVR if a linear model like linear regression doesn’t fit very well your data. This would mean you are dealing with a non linear problem, where your data is not linearly distributed. Therefore in that case SVR could be a much better solution

![svr](https://i.imgur.com/9GlOzLC.png)


## Decision Tree Regression

Decision tree builds regression or classification models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with **decision nodes** and **leaf nodes**. 

A decision node has two or more branches, each representing values for the attribute tested. Leaf node represents a decision on the numerical target. The topmost decision node in a tree which corresponds to the best predictor called **root node**. Decision trees can handle both categorical and numerical data. 

![dtr](https://i.imgur.com/u4unj7u.png)

**How does the algorithm split the data points?**

It uses reduction of standard deviation of the predictions. In other words, the standard deviation is decreased right after a split. Hence, building a decision tree is all about finding the attribute that returns the highest standard deviation reduction (i.e., the most homogeneous branches)

![dtr](https://i.imgur.com/KuBsA7g.png)



## Random Forest Regression


Random Forest Regression is a supervised learning algorithm that uses ensemble learning method for regression. Ensemble learning method is a technique that combines predictions from multiple machine learning algorithms to make a more accurate prediction than a single model.

Every decision tree has high variance, but when we combine all of them together in parallel then the resultant variance is low as each decision tree gets perfectly trained on that particular sample data and hence the output doesn’t depend on one decision tree but multiple decision trees.

![dtr](https://miro.medium.com/max/875/1*ZFuMI_HrI3jt2Wlay73IUQ.png)

## Comparison

| Regression Model         | Pros                                                                                     | Cons                                                                                  |
|--------------------------|------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| Linear Regression        | Works on any size of dataset, gives informations about relevance of features             | The Linear Regression Assumptions                                                     |
| Polynomial Regression    | Works on any size of dataset, works very well on non linear problems                     | Need to choose the right polynomial degree for a good bias/variance tradeoff          |
| SVR                      | Easily adaptable, works very well on non linear problems, not biased by outliers         | Compulsory to apply feature scaling, not well known, more difficult to understand     |
| Decision Tree Regression | Interpretability, no need for feature scaling, works on both linear / nonlinear problems | Poor results on too small datasets, overfitting can easily occur                      |
| Random Forest Regression | Powerful and accurate, good performance on many problems, including non linear           | No interpretability, overfitting can easily occur, need to choose the number of trees |


## Related Repositories

### [Data Preprocessing](https://github.com/anupam215769/Data-Preprocessing-ML)

## Credit

**Coded By**

[Anupam Verma](https://github.com/anupam215769)

<a href="https://github.com/anupam215769/Regression-ML/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=anupam215769/Regression-ML" />
</a>

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anupam-verma-383855223/)
