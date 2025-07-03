# Prediction of Any Stock Closing Price

*Graham Haun*

## Table of Contents
1) [Overview](#Overview) 
2) [Data Dictionary](<#Data Dictionary>)
3) [Requirements](#Requirements)
4) [Executive Summary](<#Executive Summary>)
    1) [Purpose](<#Purpose>)
    2) [Data Handling](<#Data Handling>)
    3) [Analysis](#Analysis)
    4) [Findings and Implications](<#Findings and Implications>)
    5) [Next Steps](#Next-Steps)
  
## Overview
A widely cited claim of investors is that '95% of all traders fail'. This claim is not back by any research but is widely agreed upon to be true in investment circles. Due to this claim or more likely the desire to not lose all their money [more than 120 million Americans rely on mutual funds and ETFs, with 68.7 million U.S. households (52.3%) owning mutual funds in 2023](https://www.ici.org/news-release/23-news-mutual-funds). Most Americans engage in some way with a [financial advisor (59%)](https://pressroom.aboutschwab.com/press-releases/press-release/2024/2024-Schwab-Modern-Wealth-Survey-Shows-Increasing-Financial-Confidence-From-Generation-to-Generation-and-Younger-Americans-Investing-at-an-Earlier-Age/default.aspx). These financial adviors and investment firms typically do not beat the market. Most firms over the past decade have underperformed compared to the S&P500 with only about [10-15% outperforming it](https://portfolio-adviser.com/the-15-of-us-funds-that-beat-the-sp-500-over-the-past-decade/). Most firms have begun to utilize machine learning to help give them an edge by making their clients more money.

Herein an random forest model was deployed to predict the closing price of a stock on the US stock market. This prediction was also extended out to 7 days and will be able to assist investment firms in making the decision of when to sell a stock and when to purchase a stock for their clients.

## Data Dictionary
Stock data and news for that stock was collected using [Alpha Vantage](https://www.alphavantage.co/). The news data consisted of the article, title, and sentiment while the stock data included the opening and closing price along with volume.

Stock Data
| Information | Data Type | Description | Notes |
|---|---|---|---|
| `date` | `string` | Date on which the data was collected. | Converted to a `datetime` object for time series analysis. |
| `open` | `float` | Opening price of the stock. |  |
| `high` | `float` | Highest price observed for the stock that day. |  |
| `low` | `float` | Lowest price observed for the stock that day. |  |
| `close` | `float` | Closing price for the stock that day. |  |
| `volume` | `float` | Amount of shares bought and sold for the stock that day. |  |

Stock News Data
| Information | Data Type | Description | Notes |
|---|---|---|---|
| `overall_sentiment_score` | `float` | A numerical value given for the overall sentiment of the article. | Negative being bearish and positive being bullish. |
| `overall_sentiment_label` | `float` | Categorical label corresponding to the overall sentiment score. | This was originally in terms of bearish or bullish but was changed to numbers for the model. |
| `ticker_relevance_score` | `float` | How relevant the article is to the chosen stock. |  |
| `ticker_sentiment_score` | `float` | The sentiment for the ticker in the same scale as the overall sentiment score. | |
| `ticker_sentiment_label` | `float` | Similar to overall sentiment label but for the chosen stock. | This was originally in terms of bearish or bullish but was changed to numbers for the model. |

### Hardware
The random forest model used was loaded in using params already discovered during the testing phase. To run the testing phase n_iter 300 was used which took about 3-5 minutes to run using an i9 processor. Below are the libraries used to run the streamlit app. Downloading the necessary libraries for streamlit is essential for this app to work.
### Software
| Library | Module | Purpose |
| --- | --- | --- |
| `pandas` || Read our data into a DataFrame, clean it, engineer new features, and write it out to submission files.|
| `sklearn` | `ensemble`| `RandomForestRegressor` for random forest model.|
| `datemtime` | `datetime`| Used to get current date and work with dates.|
| `pathlib` | `Path` | Used to work with file and directory paths.|
| `os` | | Access operating level commands within python.|
| `joblib` | | Used for saving and loading machine learning models in python.|
| `plotly.graph_objects` | | Create interactive plots and charts.|
| `requests` | | Used to make HTTP requests to web APIs.|
| `streamlit` | | Required for making a streamlit app.|
| `dotenv` | `load_dotenv` | Loads environment variables from a .env file to keep the contents secret. Used to hide the API key.|
| `pandas.tseries.offsets` | `BDay` | Used for date calculations that skip weekends and holidays.|

### Data Handling
Sentiment labels were changed into numerical values where bearish, somewhat-bearish, neutral, somewhat-bullish, and bullish were changed to 1–5 respectively. Dropped columns included title, topics, authors, summary, and source. The rest of the dataframe was then changed to a float value to better analyze for the model. 

### Analysis
Initial analysis consisted on using an LSTM (Long Short-Term Memory) model but the model failed to accurately predict any stock price and would activly predict the price continuing to go up or down over the 7 days. The model was then changed to a random forest model where a pipeline was used to initiate a random search CV and from there those paramters were entered into a GridSearchCV to better narrow down the best parameters.

The random forest was able to accuratly predict the closing stock price for the 6-month period that was gathered for the stock using Alpha Vantage. This prediction was usually within $1–5 of the actual closing price. The 7 day prediction price also showed a reasonable price that did not plummet or ascend rapidly.

### Findings and Implications
The model was able to accurately predict the closing price of the chosen stock within \\$1–5. The 7 day out prediction appeared reasonable with stocks like Microsoft not dropping to \\$200 when it is $400. This model was not tested for market crashes and could incorrectly predict them so caution is advised.

Overall this close price predictor could be a good addition to the resources of investment firms. Allowing them to get an edge on the competition by generating their clients mor emoeny through accurate weekly or day trades.
### Next Steps
Future work can include setting up an automated stock trader to test the model against the market. This could be done on the various free trading sites available that do not use real money. Further work could also extend the predictions out a month of 6 months.