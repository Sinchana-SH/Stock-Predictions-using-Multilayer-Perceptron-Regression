# Stock-Predictions-using-Multilayer-Preceptron-Regression


Abstract:
Stock price prediction is a crucial aspect of financial decision-making, aiding investors in maximizing returns and managing risks in dynamic market environments. This study investigates the application of Multilayer Perceptron (MLP) regression, a type of artificial neural network, for stock price forecasting. The research encompasses a comprehensive methodology involving data preprocessing, model creation, training, evaluation, and prediction. The dataset incorporates various features extracted from OHLC charts, index prices, technical indicators, and date-related attributes. The MLP model architecture comprises five layers, including an input layer ,three hidden layers with 64-32-16-16 neurons, and an output layer of 1 neuron. Rectified Linear Unit (ReLU) activation functions are employed in the hidden layers, while the Adam optimizer optimizes model training with Mean Squared Error (MSE) as the loss function. Evaluation metrics such as MSE and R-squared are utilized to assess model performance on training, validation, and test sets, with visualization aiding in interpretation. The outcomes demonstrate promising accuracy in stock price prediction, emphasizing the potential of MLP-based approaches for informed decision-making in financial markets. The study contributes insights into the efficacy of neural networks in capturing complex market dynamics and underscores the significance of continuous refinement and validation for improving prediction accuracy and reliability. Overall, the research highlights the value of MLP regression in stock price forecasting and offers valuable implications for future developments in machine learning and finance.


Introduction:
   
In the dynamic landscape of financial markets, the ability to forecast stock prices accurately has become an indispensable tool for investors, traders, and financial analysts. As traditional methods prove to be insufficient in capturing the intricacies of market trends, advanced machine learning techniques have emerged as powerful tools for predicting stock prices with greater precision. Among these methodologies, Multilayer Perceptron (MLP) regression has garnered significant attention for its capacity to model complex relationships within financial datasets.

This report delves into the realm of stock prediction using Multilayer Perceptron Regression, an artificial neural network architecture that holds promise in deciphering intricate patterns and nonlinear dependencies in stock market data. By leveraging the capabilities of MLP regression, we aim to unravel hidden insights and enhance the accuracy of stock price forecasts, thereby empowering market participants to make more informed and strategic investment decisions.

Throughout this report, we will explore the theoretical foundations of Multilayer Perceptron Regression, discuss its application in the context of stock prediction, and present empirical findings based on real-world financial data. By blending theoretical insights with practical applications, this report aims to contribute to the evolving landscape of financial forecasting, offering a comprehensive understanding of the potential and limitations of MLP regression in the realm of stock market predictions. As we navigate through the intricacies of this innovative approach, we strive to provide valuable insights that can empower investors and financial professionals to navigate the ever-changing landscape of financial markets.


1.Literature review:
(Talal Alotaibi et al., 2018) One of the research papers investigated the application of artificial neural networks, specifically using backpropagation, for stock market prediction in the Saudi market. The study utilized real datasets from the Saudi Stock Exchange (TADAWUL) and historical oil prices, aiming to assess the effectiveness of neural network methods in forecasting stock movements. The findings highlight the capability of backpropagation-trained neural networks in predicting stock exchange movements in the less-explored Saudi market.

(Yogitha Deshmukh, 2019) addresses the challenges of the dynamic market economy by advocating for a Machine Learning approach in stock market prediction. The focus is on using Multilayer Perceptron (MLP) and neural networks to predict various aspects of stocks or indices, aiming to empower investors with accurate predictions for informed decision-making in the rapidly changing stock market landscape.

(Özgür İcan et al., 2017) utilizes artificial neural networks for stock prediction, emphasizing the importance of directional prediction accuracy and model profitability as benchmarks over traditional forecast error measures. The paper critiques measures like mean absolute deviation and root mean squared error for their perceived inability to showcase the practical usefulness of prediction models in terms of financial gains. The review evaluates 25 papers meeting specified criteria and presents a summary in a table format to contribute insights for the development and evaluation of artificial neural network-based stock prediction models.
2.Methodology 
2.1 Data Description :
The dataset has around 60 features which includes features extracted from OHLC (Open-high-low-close chart), other index prices such as QQQ(Nasdaq-100 ETF) & S&P 500, technical Indicators such as Bollinger bands, EMA(Exponential Moving Averages, Stochastic %K oscillator, RSI etc).Furthermore, it's been created to lagged features from previous day price data as previous day prices affect the future stock price.Then, the data has date features which specify, if it's a leap year, if its month start or end, Quarter start or end, etc.
All of these features have something to offer for forecasting. Some tells us about the trend, some gives us a signal if the stock is overbought or oversold, some portrays the strength of the price trend.

![image](https://github.com/Sinchana-SH/Stock-Predictions-using-Multilayer-Preceptron-Regression/assets/116704673/c88bd92d-1a7c-418f-ad13-990458be53be)
Figure 1.0 : Dataset.head()

 
![image](https://github.com/Sinchana-SH/Stock-Predictions-using-Multilayer-Preceptron-Regression/assets/116704673/947e537c-722b-47ee-81ea-267c8c5fd8e7)

Figure: 1.1 Plot of Price vs Time of the Dataset
3. Model Creation and Training:
3.1. Neural Network Architecture:
●	The model architecture is defined using TensorFlow/Keras, consisting of multiple densely connected layers.
●	MLP Model is made up of 5 layers, one Input Layer, one Output Layer, three Hidden Layers (32-16-16) all layers use ReLU as Activation Function.
●	Hidden layers contain multiple neurons, with the number and size of layers customizable based on the complexity of the problem.

3.2 . Activation Functions:
Rectified Linear Unit (ReLU) activation functions are utilized in the hidden layers to introduce non-linearity and facilitate learning complex patterns in the data.
ReLU is chosen for its simplicity and effectiveness in preventing the vanishing gradient problem.

3.3 Model Compilation:
●	The model is compiled with the Adam optimizer, a popular choice for training deep neural networks due to its efficiency and effectiveness.
●	Mean Squared Error (MSE) is selected as the loss function, suitable for regression tasks like 
●	predicting stock prices.
●	Optionally, additional evaluation metrics such as Mean Absolute Error (MAE) or R-squared can be specified to monitor model performance during training.

3.4 Training Procedure:
The training dataset is prepared, consisting of input features and target values (stock prices).A separate validation dataset is used to monitor the model's performance and prevent overfitting. The model is trained over multiple epochs, with each epoch involving one pass through the entire training dataset. Batch size is specified to determine the number of samples processed before updating the model's parameters, balancing computational efficiency and memory usage.
Training progress is monitored, including training and validation loss, to assess model performance and detect overfitting. Early stopping is employed based on validation performance to prevent the model from continuing to train if validation loss does not improve.

 3.5 Hyperparameter Tuning:
Hyperparameters such as learning rate, batch size, and the number of neurons per layer are adjusted to optimize model performance. Techniques like grid search or random search are used to efficiently search the hyperparameter space and identify optimal values.

 4.0 Model Evaluation :
4.1 Training Performance:
During model training, the performance is monitored using the `history` object returned by the `fit()` method. This object contains metrics such as training loss (MSE) and validation loss. From the training and validation loss over epochs, insights can be gained into how well the model fits the training data and whether it is overfitting or underfitting.

4.2 Validation Performance:
The `evaluate_model()` function computes and prints the mean squared error (MSE) and R-squared for the training, validation, and test sets.
This provides insights into the model's performance on both seen and unseen data, helping to assess its generalization ability.



4.3 Interpretation:
Interpretation of evaluation metrics such as MSE and R2 helps to understand the discrepancy between predicted and actual stock prices.
Lower MSE values indicate better prediction accuracy, while higher R2 values signify a stronger correlation between predicted and actual prices. The MSE AND R^2 for our model can be analyzed through Figure 1.2.

4.4 Iterative Improvement:
Model evaluation is an iterative process that involves refining the model architecture, hyperparameters, and input features based on performance feedback. Continuous monitoring and evaluation of model performance facilitate ongoing improvement and optimization, aiming to enhance prediction accuracy and robustness.


5.0 PREDICTION:
5.1. Training Set Evaluation:
The model predictions (Y_train_pred) on the training set are compared with the actual                          target values (Y_train).
This is used  in assessing how well the model fits the training data.

5.2. Validation Set Evaluation:
The model predictions (Y_val_pred) on the validation set are compared with the actual target values (Y_val).This helps in evaluating the generalization performance of the model on unseen data.It helps in tuning hyperparameters and preventing overfitting.

5.3. Test Set Evaluation:
The model predictions (Y_test_pred) on the test set are compared with the actual target         values (Y_test).This provides an unbiased estimate of the model's performance on completely unseen data.It helps in assessing how well the model will perform in real-world scenarios.

6.0 Interpretation of Results :
6.1 Training Set Performance:
A high level of agreement between predicted and actual values on the training set indicates that the model has learned the underlying patterns well. However, excessively high performance on the training set may suggest overfitting.The model is trained using the ‘fit’ method, specifying the training data, validation data, number of epochs, and batch size.The training process iterates over multiple epochs, adjusting the model's parameters to minimize the loss function.
Training MSE: 6.2007
Training R2: 0.9990

6.2 Validation Set Performance:
After training, the model's performance is evaluated using various metrics, including MSE (Mean Squared Error) and R2.These metrics provide insights into the model's accuracy and predictive capabilities on both training and validation sets.
The performance on the validation set helps in determining if the model has generalized well to unseen data. A similar performance on both training and validation sets indicates good generalization.
Validation MSE: 4.9485
Validation R2: 0.99910

6.3 Test Set Performance:
The Multilayer Perceptron Regression model exhibits impressive test set performance, reflected in a low Mean Squared Error (MSE) of 3.96 and a high R-squared value of 0.9994. These metrics underscore the model's accuracy and robustness in predicting stock prices, reinforcing its efficacy for real-world applications.

6.3 Visualization of Training Progress:
The loss function's values on the training, test and validation sets are plotted against the number of epochs as shown in Figure 1.3 and MSE plot in Figure 1.4. This visualization helps in assessing the model's convergence and identifying potential overfitting or underfitting issues.

 ![image](https://github.com/Sinchana-SH/Stock-Predictions-using-Multilayer-Preceptron-Regression/assets/116704673/3a1f489a-096b-4ab8-823e-150531e5af39)
![image](https://github.com/Sinchana-SH/Stock-Predictions-using-Multilayer-Preceptron-Regression/assets/116704673/9089d0d2-8673-40f9-af2e-2801d12a2336)

Figure 1.3 Plot of Actual vs Predicted values and MSE AND R ^2 values for Train test and validation sets


 

The performance metrics for the Multilayer Perceptron Regression model in stock prediction showcase remarkable accuracy and robustness. The Mean Squared Error (MSE) values for training, validation, and test datasets are 6.20, 4.95, and 3.96, respectively, indicating minimal prediction errors. Moreover, the R-squared values for training (0.999), validation (0.9991), and test (0.9994) demonstrate an exceptional fit of the model to the actual stock price data. These results underscore the efficacy of the Multilayer Perceptron Regression in capturing complex patterns within financial datasets, making it a promising tool for precise and reliable stock price predictions . Overall, the outcomes underscore the value of employing MLP-based approaches in stock price prediction for informed decision-making and strategy development.

![image](https://github.com/Sinchana-SH/Stock-Predictions-using-Multilayer-Preceptron-Regression/assets/116704673/5b91c248-d4ae-488a-835d-fb9ff9cb094b)

Figure 1.4 MSE plot for Training , Test and Validation sets


CONCLUSION

In conclusion, the utilization of Multilayer Perceptron (MLP) for stock prediction has proven to be a valuable and insightful approach in this project. The thorough implementation of the methodology, from data collection and preprocessing to model training and evaluation, has provided a structured and effective framework for forecasting stock prices. The MLP model, configured with careful consideration of architecture and optimized through rigorous training iterations, has demonstrated its ability to capture complex patterns within historical stock market data. The project's success lies in the systematic methodology employed, ensuring the reliability and accuracy of the stock predictions generated by the MLP model. The findings contribute not only to the field of financial forecasting but also emphasize the significance of artificial intelligence in enhancing decision-making processes in the dynamic realm of stock markets. Overall, the project underscores the potential of MLP in stock prediction and offers valuable insights for future developments in the intersection of machine learning and finance.



REFERENCE 

1.	"Indian Stock-Market Prediction using Stacked LSTM AND Multi-Layered Perceptron." Available:[https://www.academia.edu/66903471/Indian_Stock_Market_Prediction_using_Stacked_LSTM_AND_Multi_Layered_Perceptron].

2.	 R. Arjun Raj, "Stock Market Index Prediction Using Multilayer Perceptron and Long Short Term Memory Networks: A Case Study on BSE Sensex," 2018.

3.	 N. Rouf, M.B. Malik, T. Arif, S. Sharma, S. Singh, S. Aich, H.-C. Kim, "Stock Market Prediction Using Machine Learning Techniques: A Decade Survey on Methodologies, Recent Developments, and Future Directions," Electronics, vol. 10, no. 21, p. 2717, 2021. [Online]. Available: [https://doi.org/10.3390/electronics10212717].

4.	 Y. Deshmukh, "Stock Market Prediction using Neural Network and MLP," International Journal of Innovative Science, Engineering & Technology, vol. 7, no. 2, Feb. 2020. [Online]. Available: [www.ijiset.com].

5.	A. Namdari and T.S. Durrani, "A Multilayer Feedforward Perceptron Model in Neural Networks for Predicting Stock Market Short‑term Trends," 2021. [Online]. Available: [DOI not available].

6.	R. Arjun Raj, "Stock Market Index Prediction Using Multilayer Perceptron and Long Short Term Memory Networks: A Case Study on BSE Sensex," International Journal of Research and Scientific Innovation (IJRSI), vol. 5, no. 7, Jul. 2018. [Online]. Available: [ISSN 2321–2705].

7.	 M.L. Sherin Beevi and C.A. Daphine Desona Clemency, "STOCK FEE PREDICTION USING MULTI -LAYER PERCEPTRON," International Research Journal of Engineering and Technology (IRJET), vol. 08, no. 02, Feb. 2021. [Online]. Available: [www.irjet.net].

