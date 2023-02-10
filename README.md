# Server Usage Prediction Time Series
## Summary
The data consisted of daily HDD, RAM, and CPU usage requests. The data originated from BRI, which was restricted. Multiple columns are numerical and categorical, but only numerical ones are used. Numerical values are unstandardized and categorical data are not encoded. Although the data is daily, it sometimes skips every once in a while due to no request
## Objective
The objective is to determine how many HDD, RAM, and CPU will be used in the future based on prior daily server use.
## Problem Definition
BRI needs server usage prediction to prepare for future use. Instead of wasting money on an excessive amount of funds for procurement, The predictions can justify the exact amount of servers needed for future procurement. In the long run, this will help the bank saves money and be more efficient in funding procurement.
## Steps
1. Resample from daily to monthly
2. EDA(Seasonal Decomposition)
3. Outlier Detection
4. Replace outliers with median
5. Vanilla Modeling with Holtwinters
6. Evaluate with Mean Absolute Percentage Error
7. Make new feature with lagging
8. modeling with XGBoost
6. Evaluate with Mean Absolute Percentage Error
## Results
Model able to predict server Usage three months ahead. The MAPE of prediction can reach almost 90% with XGBoost and using lagging

 

 
