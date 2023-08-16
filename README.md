# Regression

Regression is a [Supervised Learning](https://en.wikipedia.org/wiki/Supervised_learning#:~:text=Supervised%20learning%20(SL)%20is%20the,on%20example%20input%2Doutput%20pairs.&text=A%20supervised%20learning%20algorithm%20analyzes,used%20for%20mapping%20new%20examples.) technique which helps in finding the correlation between variables and enables us to predict the continuous output variable based on the one or more predictor variables.

In Regression, we plot a graph between the variables which best fits the given datapoints, using this plot, the machine learning model can make predictions about the data. In simple words, "***Regression shows a line or curve that passes through all the datapoints on target-predictor graph in such a way that the vertical distance between the datapoints and the regression line is minimum.***" The distance between datapoints and line tells whether a model has captured a strong relationship or not.

### Simple Linear Regression [Code](https://github.com/anupam215769/Regression-ML/blob/main/Simple%20Linear%20Regression/Simple-Linear-Regression.ipynb) OR <a href="https://colab.research.google.com/github/anupam215769/Regression-ML/blob/main/Simple%20Linear%20Regression/Simple-Linear-Regression.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

### Multiple Linear Regression [Code](https://github.com/anupam215769/Regression-ML/blob/main/Multiple%20Linear%20Regression/multiple_linear_regression.ipynb) OR <a href="https://colab.research.google.com/github/anupam215769/Regression-ML/blob/main/Multiple%20Linear%20Regression/multiple_linear_regression.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

### Polynomial Regression [Code](https://github.com/anupam215769/Regression-ML/blob/main/Polynomial%20Regression/polynomial_regression.ipynb) OR <a href="https://colab.research.google.com/github/anupam215769/Regression-ML/blob/main/Polynomial%20Regression/polynomial_regression.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

### Support Vector Regression (SVR) [Code](https://github.com/anupam215769/Regression-ML/blob/main/Support%20Vector%20Regression%20(SVR)/support_vector_regression.ipynb) OR <a href="https://colab.research.google.com/github/anupam215769/Regression-ML/blob/main/Support%20Vector%20Regression%20(SVR)/support_vector_regression.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

### Decision Tree Regression [Code](https://github.com/anupam215769/Regression-ML/blob/main/Decision%20Tree%20Regression/decision_tree_regression.ipynb) OR <a href="https://colab.research.google.com/github/anupam215769/Regression-ML/blob/main/Decision%20Tree%20Regression/decision_tree_regression.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

### Random Forest Regression [Code](https://github.com/anupam215769/Regression-ML/blob/main/Random%20Forest%20Regression/random_forest_regression.ipynb) OR <a href="https://colab.research.google.com/github/anupam215769/Regression-ML/blob/main/Random%20Forest%20Regression/random_forest_regression.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

>Don't forget to add Required Data files in colab. Otherwise it won't work.

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

```
g(x) = f0(x) + f1(x) + f2(x) + .... + fn(x)
```

![dtr](https://miro.medium.com/max/875/1*ZFuMI_HrI3jt2Wlay73IUQ.png)


## Evaluating Regression Models Performance

### R-squared

R-squared statistic or coefficient of determination is a scale invariant statistic that gives the proportion of variation in target variable explained by the linear regression model.

- **Total Sum of Squares** - Total variation in target variable is the sum of squares of the difference between the actual values and their mean.

![formula](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/07/TSSchange.png)

- **Residual Sum of Squares** - RSS gives us the total square of the distance of actual points from the regression line.

- **Calculate R-Squared** - 

          R-squared = (TSS-RSS)/TSS

                    = Explained variation/ Total variation

                    = 1 – Unexplained variation/ Total variation
                    
                    
If we have a really low RSS value, it would mean that the regression line was very close to the actual points. This means the independent variables explain the majority of variation in the target variable. In such a case, we would have a really high R-squared value.

![for](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/07/R2-decrease.png)

On the contrary, if we have a really high RSS value, it would mean that the regression line was far away from the actual points. Thus, independent variables fail to explain the majority of variation in the target variable. This would give us a really low R-squared value.

![for](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/07/R2-increase.png)

So, this explains why the R-squared value gives us the variation in the target variable given by the variation in independent variables.

#### Problems with R-squared statistic

The R-squared statistic isn’t perfect. In fact, it suffers from a major flaw. Its value never decreases no matter the number of variables we add to our regression model. That is, even if we are adding redundant variables to the data, the value of R-squared does not decrease. It either remains the same or increases with the addition of new independent variables. This clearly does not make sense because some of the independent variables might not be useful in determining the target variable. Adjusted R-squared deals with this issue.


### Adjusted R-squared

The Adjusted R-squared takes into account the number of independent variables used for predicting the target variable. In doing so, we can determine whether adding new variables to the model actually increases the model fit.

![adjusted](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/07/edit.png)

Here,

- **n** represents the number of data points in our dataset
- **k** represents the number of independent variables, and
- **R** represents the R-squared values determined by the model.


So, if R-squared does not increase significantly on the addition of a new independent variable, then the value of Adjusted R-squared will actually decrease.

![adj](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/07/edit1.png)

On the other hand, if on adding the new independent variable we see a significant increase in R-squared value, then the Adjusted R-squared value will also increase.

![ad](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/07/edit2.png)


We can see the difference between R-squared and Adjusted R-squared values if we add a random independent variable to our model.

![result](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/07/result.png)

As you can see, adding a random independent variable did not help in explaining the variation in the target variable. Our R-squared value remains the same. Thus, giving us a false indication that this variable might be helpful in predicting the output. However, the Adjusted R-squared value decreased which indicated that this new variable is actually not capturing the trend in the target variable.


#### Code

```
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
```




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


### [Classification](https://github.com/anupam215769/Classification-ML)

### [Clustering](https://github.com/anupam215769/Clustering-ML)

### [Association Rule Learning](https://github.com/anupam215769/Association-Rule-Learning-ML)

### [Reinforcement Learning](https://github.com/anupam215769/Reinforcement-Learning-ML)

