# TESLA-Stock-prediction

ABSTRACT

Time series modeling and analysis are critical in a variety of fields, including economics, engineering, environmental research, and social science. For scientists and academic researchers, choosing the optimum time series model with appropriate parameters in forecasting is a difficult task. To increase the accuracy of modeling and predicting time series, hybrid models integrating neural networks and traditional Autoregressive Moving Average (ARMA) models are being applied. Information-theoretic techniques, such as AIC, BIC, and HQ, are used to choose the majority of existing time series models. A model selection strategy based on Random Forest Regression, long short-term memory (LSTM), and Autoregressive Moving Average (ARMA) is revisited in this study.

INTRODUCTION

Time series exploration is a crucial field of data analysis that involves collecting knowledge from previous observations in order to determine the evolution of a phenomenon in the present and make it easier to project into the future. This investigation also defines some qualities necessary to comprehend the current condition of this phenomenon and highlights certain correlations that may appear inevitable to the naked eye. The scientific community has put in a lot of work to improve time series exploration and analysis.
The ARIMA model, developed by Box and Jenkins in 1976, is one of the most well-known and widely used models for this type of analysis. This model's success stems mostly from its adaptability, quality, and versatility. It can be used with autoregressive processes (AR), moving average processes (MA), or a mix of both (ARMA). ARIMA works with "stationary" processes, but it can adapt to processes that aren't in its integrated form (the "I" in ARIMA). Last but not least, the SARIMA time series may show seasonal changes in this model. The significant error rate of this model when normalcy and/or linearity in the data are lost is a serious flaw. Recent advancements in information technology have aided in the generation of large amounts of data on the one hand, and the preparation of necessary IT support for computations on the other.
This advancement has also resulted in the deployment of more efficient learning models than those offered by traditional statistics. Then, thanks to deep learning, neural networks achieve their peak and surpass standard models, whether it's regression, classification, or more advanced tasks like multimedia data processing (image, sound). The ability of neural networks to adapt to the properties provided by the data itself is one of their distinguishing characteristics. Despite the loss of certain important aspects of traditional statistics such as linearity, normality, homoscedasticity, and observation independence, this data-oriented approach improves learning. Neural networks, like the ARIMA model, come in a variety of designs, each matching to a specific problem (RNN, CNN, DQN, GAN, and so on).

In this paper we will use Regression Techniques such as ARIMA, LSTM and Random Forest along with Time Series Analysis and compare Accuracy, RMSE values to pick the Best Model.

MOTIVATION

Artificial Intelligence (AI) has been a popular technology in recent years, with applications including driverless automobiles, intelligent robotics, picture and speech recognition, automatic translations, and medical assistants. As a result, stock market forecasting in the light of AI and using various machine learning algorithms has become a significant issue in the financial and economic domains. As a result, throughout the last decade, researchers have been thinking about developing trustworthy predictive models. The need to better predict stock values has prompted researchers to work on improving current predictive machine learning models. The rationale for this is that shareholders and investors have the freedom to develop plans and strategic methods to making investment and future activity decisions. As a result, businesses and people seek out any predictive strategy that will help them earn more money from the stock market with less risk. Because of the unpredictable characteristics and intricate connections of the stock market, forecasting stock market is regarded one of the most challenging problems in finance. Because of the unpredictability of the stock market, there are no specific machine learning models that can accurately anticipate the stock market, and there is still more work to be done in this sector, which motivates us to investigate and develop a better prediction system. In recent years, various machine learning approaches have been used to forecast the stock market. Among them models such as ARIMA, LSTM, Random Forest and so on had been exploited to improve time series forecasting.

PROBLEM STATEMENT

Stock Prices can be considered as Random Variables. In Time Series Analysis we have considered the Stock Prizes as Time series data to predict the stock price of Tesla Stock.
In mathematics, a time series is a series of data points indexed (or listed or graphed) in time order. Most commonly, a time series is a sequence taken at successive equally spaced points in time. Thus it is a sequence of discrete-time data. Methods for studying time series data in order to extract relevant statistics and other data features are referred to as time series analysis. The employment of a model to predict future values based on previously observed values is known as time series forecasting. While regression analysis is frequently used to assess links between one or more separate time series, it is not always referred to as "time series analysis," which refers to relationships between different points in time within a single series. Interrupted time series analysis is used to find changes in a time series' evolution from before to after some action that may alter the underlying variable. For Stock price prediction we have considered the stock price Data From previous 5 years to predict the stock price of next day. For this project we have considered ARIMA, LSTM and Random Forest Approaches’ in Time Series Analysis.



 

LSTM 

The architecture of an LSTM model is more complex to present. In this document, we will simply present an overview diagram as well as the equations that govern the set and a quick description. The Long Short-Term Memory (LSTM) network is the most widely used architecture in practice to address the problem of gradient disappearance. This network structure was proposed by Sepp Hochreiter and Jrgen Schmidhuber in 1997 [10]. The idea associated with the LSTM is that each computational unit is linked not only to a hidden state h but also to a state c of the cell that plays the role of memory. The change from ct−1 to ct is done by constant gain transfer equal to 1, so that errors are propagated at previous steps without any gradient disappearance phenomenon. The status of the cell can be modified through a door that allows or blocks the update (input gate). Similarly, a door controls whether the state of the cell is communicated at the output gate of the LSTM unit. The most common version of LSTMs also uses a gate to reset the cell state (forget gate). The architecture of LSTM is as below: -
 
The first step in our LSTM is to decide what information we’re going to throw away from the cell state. This decision is made by a sigmoid layer called the “forget gate layer.” It looks at ht−1ht−1 and xt, and outputs a number between 00 and 11 for each number in the cell state Ct−1Ct−1. A 11 represents “completely keep this” while a 00 represents “completely get rid of this.”
 
The next step is to decide what new information we’re going to store in the cell state. This has two parts. First, a sigmoid layer called the “input gate layer” decides which values we’ll update. Next, a tanh layer creates a vector of new candidate values, C~t, that could be added to the state. In the next step, we’ll combine these two to create an update to the state.
 
It’s now time to update the old cell state, Ct−1Ct−1, into the new cell state Ct. The previous steps already decided what to do, we just need to actually do it.
We multiply the old state by ft, forgetting the things we decided to forget earlier. Then we add it∗C~tit∗C~t. This is the new candidate values, scaled by how much we decided to update each state value.
 
Finally, we need to decide what we’re going to output. This output will be based on our cell state, but will be a filtered version. First, we run a sigmoid layer which decides what parts of the cell state we’re going to output. Then, we put the cell state through tanh (to push the values to be between −1−1 and 11) and multiply it by the output of the sigmoid gate, so that we only output the parts we decided to.
 
With the advent of deep architectures, LSTM is widely used today in sequence learning and has demonstrated great experimental power [11]. However, this architecture is subject to many changes, it has almost as many variations as the papers that use it.
LSTM Parameter Modification: -
LSTM has 4 Feed Forward Neural Networks (FFNN). Let g = No. of FFNN’s, h = hidden layers, i = dimension of input
So we have, number of parameters = g * [ h * (h + i) + h ]
We have following parameter values – 
1.	Batch size – how many samples in each batch, after which the weights will be modified.
2.	Timestep – how many values in sequence to consider
3.	Features – numbers of features form a particular record
4.	Unit – dimension of the cell state and hidden state
5.	Epoch – number of iterations the model will run on the entire dataset
6.	Optimizer – optimizer to reduce the loss function, Adam
7.	Loss Function – how the actual value is compared against predicted value, mean squared error
8.	Dropout Ratio – how many of the previous nodes output to be dropped before passing it to next layer.
Out of the above, timestep of the input does not play much role in the number of parameters. This is because of the inner weights W & U and biases b is used same across the timesteps.
We have run the model using different parameter values for hidden layer, timestep, batch size, units, epoch and tried running the model to predict the Tesla stock price. There was a lot of difference in the model behavior after changing these parameters.

ARIMA

EXPONENTIAL SMOOTHING:

Exponential smoothing was proposed in the late 1950s (Brown, 1959; Holt, 1957; Winters, 1960), and has motivated some of the most successful forecasting methods. Forecasts produced using exponential smoothing methods are weighted averages of past observations, with the weights decaying exponentially as the observations get older. In other words, the more recent the observation the higher the associated weight. This framework generates reliable forecasts quickly and for a wide range of time series, which is a great advantage and of major importance to applications in industry.

ARIMA:

ARIMA is specially designed for Time series analysis. 
Auto Regressive (AR): In a multiple regression model, we forecast the variable of interest using a linear combination of predictors. In an autoregression model, we forecast the variable of interest using a linear combination of past values of the variable. The term autoregression indicates that it is a regression of the variable against itself.
Thus, an autoregressive model of order pp can be written as, 
 yt=c+ϕ1yt−1+ϕ2yt−2+⋯+ϕpyt−p+εt,
where εt is white noise. This is like a multiple regression but with lagged values of yt as predictors. We refer to this as an AR (pp) model, an autoregressive model of order pp.    

Moving Average (MA): Rather than using past values of the forecast variable in a regression, a moving average model uses past forecast errors in a regression-like model   yt=c+εt+θ1εt−1+θ2εt−2+⋯+θqεt−q, where εt is white noise. We refer to this as an MA (qq) model, a moving average model of order qq. Of course, we do not observe the values of εt, so it is not really a regression in the usual sense

Changing the parameters θ1,…,θq results in different time series patterns. As with autoregressive models, the variance of the error term εtεt will only change the scale of the series, not the patterns.
It is possible to write any stationary AR(p) model as an MA(∞) model. For example, using repeated substitution, we can demonstrate this for an 
AR(1)Model:
yt=ϕ1yt−1+εt=ϕ1(ϕ1yt−2+εt−1)+εt=ϕ21yt−2+ϕ1εt−1+εt=ϕ31yt−3+ϕ21εt−2+ϕ1εt−1+εtetc.
Provided −1<ϕ1<1, the value of ϕ1k will get smaller as k gets larger. So eventually we obtain yt=εt+ϕ1εt−1+ϕ21εt−2+ϕ31εt−3+⋯, an MA(∞∞) process.

If we combine differencing with autoregression and a moving average model, we obtain a non-seasonal ARIMA model. ARIMA is an acronym for Autoregressive Integrated Moving Average (in this context, “integration” is the reverse of differencing). The full model can be written as
 y′t=c+ϕ1y′t−1+⋯+ϕpy′t−p+θ1εt−1+⋯+θqεt−q+εt,
where yt′ is the differenced series (it may have been differenced more than once). The “predictors” on the right hand side include both lagged values of yt and lagged errors. We call this an ARIMA(p,d,q) model, where
	
p=	order of the autoregressive part;
d=	degree of first differencing involved;
q=	order of the moving average part.
The same stationarity and invertibility conditions that are used for autoregressive and moving average models also apply to an ARIMA model.

Maximum Likelihood Estimation:         
Once the model order has been identified (i.e., the values of p, d and q), we need to estimate the parameters cc, ϕ1,…,ϕp, θ1,…,θq. When R estimates the ARIMA model, it uses maximum likelihood estimation (MLE). This technique finds the values of the parameters which maximize the probability of obtaining the data that we have observed. For ARIMA models, MLE is like the least squares estimates that would be obtained by minimizing
∑εt2¬¬
In practice, R will report the value of the log likelihood of the data; that is, the logarithm of the probability of the observed data coming from the estimated model. For given values of p, d and q, R will try to maximize the log likelihood when finding parameter estimates.

Information Criteria
Akaike’s Information Criterion (AIC), which was useful in selecting predictors for regression, is also useful for determining the order of an ARIMA model. It can be written as 
AIC=−2log(L)+2(p+q+k+1),
where LL is the likelihood of the data, k=1 if c≠0 and k=0 if c=0. Note that the last term in parentheses is the number of parameters in the model (including σ2, the variance of the residuals).
For ARIMA models, the corrected AIC can be written as 
AICc=AIC+2(p+q+k+1)(p+q+k+2)T−p−q−k−2,
and the Bayesian Information Criterion can be written as
BIC=AIC+[log(T)−2](p+q+k+1)
Good models are obtained by minimizing the AIC, AICc or BIC. 

RANDOM FOREST

PREREQUISITES TO RANDOM FOREST:

DECISION TREE

A decision tree is a supervised machine learning algorithm. Though this algorithm can be used for regression and classification problems, it is mostly used for classification problems. A decision tree follows a set of if-else conditions to visualize the data and classify it according to the conditions.
In decision trees, internal nodes represent the features of a data, branches represent the decision rules and each leaf node represents the outcome.
There are 2 main types of nodes, the Decision Node and Leaf Node.
- Decision nodes are used to make any decision and have multiple branches
- Leaf nodes are the output of those decisions and do not contain any further branches



The main main strengths of decision trees are
- Itis a graphical representation for getting all the possible solutions to a problem/decision based on given conditions, this makes it interpretable.
- Decision Trees usually mimic human thinking ability while making a decision, so it is easy to understand
- The logic behind the decision tree can be easily understood due to its tree-like structure.

BAGGING

This is one of the most well-known Ensemble learning methods. For this method, a random sample of data in a training set is selected with replacement (similar to k fold cross validation). After several data samples are generated, these models are then trained independently, and the expectation goes that the average or majority of those predictions yield a more accurate estimate. This method reduces the correlation among different decision trees.

Random forest is an ensemble learning method for classification, regression and other tasks that operates by building multiple decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees. For regression tasks, the mean or average prediction of the individual trees is returned.
The random forest algorithm is an extension of the bagging method as it utilizes both bagging and feature randomness to create an uncorrelated forest of decision trees. While decision trees consider all the possible feature splits, random forests only select a subset of those features, this helps with overfitting of the data, where errors in the experiment or data collection process could perturb the values of the result.

Random forest algorithms have three main hyperparameters to be set during training. 
- Node size
- Number of trees
- Number of features sampled.

Random forest adds randomness to the model training process in two ways
- Bootstrapping → Multiple decision trees are used, which are trained using random chunks of data sampled randomly from the training data.
- Bagging - adding more diversity to the dataset and reducing the correlation among decision trees

ADVANTAGES OF RANDOM FOREST

- Low risk of Overfitting →averaging of uncorrelated trees lowers the overall variance and prediction error
- Flexibility → Can work on both Regression and Classification problems.
- Fill in missing entries → Feature bagging allows us to fill empty entries with a more reasonable accuracy
- Feature extraction → Random Forest makes it easy to evaluate variable contribution to the model

DISADVANTAGES OF RANDOM FOREST

- Needs more data →Since data is getting fragmented over multiple decision trees.
- Time consuming →due to multiple decision trees being trained
- Complex →The prediction of a single decision tree is easier to interpret when compared to a forest of them

RESULTS

Modeling on the tesla stock was a bit difficult when compared to the models generated from choosing a different stock such as a Google, Facebook, apple and many more. The reason was mainly focused on lack of data points and a haphazard behavior of the stock in recent years jumping more than 600% in just 2 years and even after that the stock was not in a more stable position. Even the distribution of the target variable which is the close variable was a right skewed distribution rather than a normal distribution. Modeling on such data does not yield a good model. 
 
 

As you can see a small glimpse of the data of the most recent data, that percentage drop/increase is quite haphazard. Predicting the next point quite accurately would be a bit difficult. To get a clearer image, I have attached the graph plotting all the points. The increase in stock price happened after 2020 and it is still behaving randomly. In recent times, the stock fell another 26% after Elon Musk acquired twitter on some irrational rumor. 
 

The modeling technique we used were mainly RANDOM FOREST, ARIMAandLSTM. These were not the only model we tried to get a more suitable result. The conventional model we tired were multi linear regression, principal component regression and support vector machines, but being a time series forecasting model, it was unable to replicate the haphazard trend generated by the stock.
We added lag solely for the purpose to capture to the trend generated by the stocks. In the diagram below, I have added the plot mapping out the trends. 

 

ARIMA

Arima or better known as Auto-Regressive Integrated Moving Average. Arima model is usually applied on a data which is stationary thus, this is not the case for us as our data was far from being stationary. We used the package ‘forecast’ to generate Arima model as it has a built-in function to build the model setting to default and choosing the engine Auto_Arima. We split the data using different time series data splitting function, choosing different percentage of test every time. The table below shows all the ways we tired and finally choosing one. 

Splitting ratio (test data)	Rmse	R-squared
8 weeks	1.2	18%
12 weeks	3.5	5%
4 weeks	5.1	0%

 As you can see from the above table that changing the ratio of the test data will bring the accuracy of the model to zero and increase the rmse value. I have highlighted the value chosen to display and work more to improve it but turns out the value highlighted was only generated by a coincidence rather than producing it. Mostly the model was giving a zero percentage of accuracy thus showing that it is unable to capture the trend. I ran the simulation about 10-20 times and 90% of the times the data produced zero accuracy. 

RANDOM FOREST
Random Forest is a popular and effective ensemble machine learning algorithm. It is widely used for classification and regression predictive modeling problems with structured (tabular) data sets. We used the package ‘forecast’ to generate random forest model as it has a built-in function to build the model setting to default and choosing the engine rangers and setting the mode regression rather than classification. We split the data using different time series data splitting function, choosing different percentage of test every time. The table below shows all the ways we tired and finally choosing one.

Splitting ratio (test)	Rmse	R squared
12 weeks	8.5	5%
4 weeks	8.1	0%
8 weeks	3.6	12%

As you can see from the above table that changing the ratio of the test data will bring the accuracy of the model to zero and increase the rmse value. I have highlighted the value chosen to display and work more to improve it but turns out the value highlighted was only generated by a coincidence rather than producing it. Mostly the model was giving a zero percentage of accuracy thus showing that it is unable to capture the trend. I ran the simulation about 10-20 times and 90% of the times the data produced zero accuracy.

As you can see that both models did not perform up to par and none of the were able to grasp the trend produced by the stock thus giving us no benefit in running the on an unseen data. I have attached the graph mapping out the trends of the actual data versus the predicted value of both the random forest and Arima:
 

I did a further analysis on both random forest and Arima to refit the testing data as well in the training data and then retransforming it back to its original value and predicting the future unseen values. The predictions are highly misleading as they did not predict well. 

 

LSTM

The last model we chose to model on was LSTM which stands for long-short-term memory. It is special kind of recurrent neural network that is capable of learning long term dependencies in data.LSTM is a deep learning model and we tired to implement it on R but was unable to do so. We implemented LSTM on python. Creating an LSTM model was a bit different than building Random Forest or Arima. LSTM was able to capture the trend if not accurately. The forecast days here were also the same 60 days. We did try to do it for 45 days as well as 30 days and even 50 days, but the best results were generated by using 60 days. We also tired changing the epoch from 25 to a bigger value like 30 or a lesser value like 20 but the result then become random. The results are below:
Epoch, prediction days	Rmse	R-squared
25, 60 	6.4	38%
30, 45	300	12%
22, 50	251	20%

As you see, the value of LSTM is more stable when compared all the other models and their iterations. LSTM was able to successfully capture the trend produced by stock as shown in the diagram below. The graph has been scaled to get a clearer image of the trend. The issue here is that we used the testing data from the training data itself, thus this is a false image of the model performance. 

We tested the model on completely unseen data and result were not good. Little part of the trend was captured but compared to others the model was the best in tracking the values. The graph shows the result of predicting on a completely unseen data. 
