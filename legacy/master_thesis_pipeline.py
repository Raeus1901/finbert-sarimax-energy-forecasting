#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# LEGACY SCRIPT — preserved for methodological transparency only.
# This file is NOT the reproducible pipeline. See notebooks/Master_Thesis.ipynb
# Google News scraper no longer functional (post-2024 anti-bot measures).
# FMP API earnings-surprises endpoint is paywalled as of 2024.
# =============================================================================
"""
Corrected & Synthesized Script
Author: Jean (Modified)
Date: January 2025
"""

import sys
sys.path.append('/Users/jean/Desktop/')  # Chemin à ajuster si nécessaire

# === Imports ===
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import time
import warnings
import matplotlib.dates as mdates

# Stationnarité, ARIMA, et tests divers
from pmdarima.arima import auto_arima
from pmdarima.arima.utils import ndiffs
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro

# Eventuels modules externes
from crawler import write_crawl_results  # suppose l’existence de ce module
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

warnings.filterwarnings('ignore')


# === 1) Paramètres généraux ===
tickers = ['FSLR', 'GE', 'NEE', 'TSLA', 'PLUG']
api_key = 'YOUR_API_KEY'  # Remplacer par votre clé Financial Modeling Prep

# === 2) Fonction pour récupérer les earnings surprises depuis l’API FMP ===
def fetch_earnings_data(ticker, api_key):
    """
    Récupère les données d'earnings surprises pour un ticker donné via l’API FinancialModelingPrep.
    """
    api_url = f"https://financialmodelingprep.com/api/v3/earnings-surprises/{ticker}?apikey={api_key}"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        earnings_data = response.json()
        earnings_df = pd.DataFrame(earnings_data)
        if not earnings_df.empty:
            earnings_df['date'] = pd.to_datetime(earnings_df['date'])
            earnings_df['earnings_surprise'] = earnings_df['actualEarningResult'] - earnings_df['estimatedEarning']
        return earnings_df
    except requests.exceptions.RequestException as err:
        print(f"Error fetching data for {ticker}: {err}")
        return pd.DataFrame()


# === 3) Tests de stationnarité (ADF et KPSS) ===
def adf_test(timeseries):
    """
    Effectue le test de Dickey-Fuller augmenté (ADF) et retourne un résumé sous forme de Series.
    """
    dftest = adfuller(timeseries.dropna(), autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
                         index=['Test Statistic', 'p-value', 
                                '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput[f'Critical Value ({key})'] = value
    return dfoutput

def kpss_test(timeseries):
    """
    Effectue le test de KPSS et retourne un résumé sous forme de Series.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        kpsstest = kpss(timeseries.dropna(), regression='c', nlags='auto')
    kpss_output = pd.Series(kpsstest[0:3], 
                            index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output[f'Critical Value ({key})'] = value
    return kpss_output


# === 4) Récupération et fusion de données (historique & earnings) pour chaque ticker ===
all_tickers_df = pd.DataFrame()  # Collecte toutes les données des tickers

for ticker in tickers:
    print(f"\n=== Fetching data for {ticker} ===")
    # Récupération des earnings
    earnings_df = fetch_earnings_data(ticker, api_key)
    
    if earnings_df is None or earnings_df.empty:
        print(f"No earnings data for {ticker}. Skipping.")
        continue
    
    # Récupération des cours boursiers
    stock_data = yf.Ticker(ticker).history(period="10y")
    stock_data.reset_index(inplace=True)
    stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.tz_localize(None)
    
    # Fusion sur la date exacte (ou inner join)
    merged_df = pd.merge(
        stock_data, 
        earnings_df, 
        left_on='Date', 
        right_on='date', 
        how='inner'
    )
    merged_df['symbol'] = ticker  # Pour distinguer le ticker dans all_tickers_df
    
    # Exemple : On affiche quelques lignes
    print(merged_df.head(3))
    
    # Concatène les données
    all_tickers_df = pd.concat([all_tickers_df, merged_df], ignore_index=True)


# === 5) Visualisation de base (Prix vs Earnings Surprise) ===
for ticker in tickers:
    subset = all_tickers_df[all_tickers_df['symbol'] == ticker]
    if subset.empty:
        continue
    
    plt.figure(figsize=(14, 7))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    sns.lineplot(data=subset, x='Date', y='Close', ax=ax1, color='blue', label=f'{ticker} Price')
    sns.lineplot(data=subset, x='Date', y='earnings_surprise', ax=ax2, color='red', label='Earnings Surprise')

    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Stock Price', fontsize=12, color='blue')
    ax2.set_ylabel('Earnings Surprise', fontsize=12, color='red')
    ax1.set_title(f'Stock Price & Earnings Surprise - {ticker}', fontsize=14)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    time.sleep(1)


# === 6) Tests de stationnarité & ARIMA / SARIMAX pour chaque ticker ===

def run_sarimax_for_ticker(ticker, df):
    """
    Pour un ticker donné, on fit un modèle SARIMAX avec la variable endogène = 'Close'
    et la variable exogène = 'earnings_surprise'. 
    """

    # Filtre les données du ticker
    subset = df[df['symbol'] == ticker].copy()
    if subset.empty:
        print(f"No data for {ticker}.")
        return

    # Variable cible et exogène
    y = subset['Close']
    X = subset['earnings_surprise'].fillna(method='ffill').fillna(method='bfill')
    
    print(f"\n=== Stationarity tests for {ticker} ===")
    # Test ADF
    adf_res = adf_test(y)
    print("ADF Test:\n", adf_res)
    # Test KPSS
    kpss_res = kpss_test(y)
    print("KPSS Test:\n", kpss_res)

    # On détermine l’ordre (p, d, q) via auto_arima
    arima_model = auto_arima(
        y, start_p=1, start_q=1, max_p=3, max_q=3,
        seasonal=False, d=None, trace=False,
        error_action='ignore', suppress_warnings=True, stepwise=True
    )
    print(f"auto_arima suggests order: {arima_model.order}")
    
    # Fit final SARIMAX avec exogène
    sarimax_model = SARIMAX(y, exog=X, order=arima_model.order,
                            enforce_stationarity=False, 
                            enforce_invertibility=False)
    sarimax_results = sarimax_model.fit(disp=False)
    print(f"\n=== SARIMAX summary for {ticker} ===")
    print(sarimax_results.summary())

    # In-sample predictions
    sarimax_predictions = sarimax_results.get_prediction(exog=X).predicted_mean

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(subset['Date'], y, color='blue', label='Actual')
    plt.plot(subset['Date'], sarimax_predictions, color='red', label='Predicted')
    plt.title(f'SARIMAX In-sample Predictions - {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    # Diagnostics : Résidus, tests, ACF
    residuals = sarimax_results.resid
    fig, ax = plt.subplots(1,2, figsize=(16, 4))
    ax[0].plot(residuals)
    ax[0].set_title('SARIMAX Residuals')
    sns.kdeplot(residuals, fill=True, ax=ax[1])
    ax[1].set_title('SARIMAX Residual Density')
    plt.show()

    # Durbin-Watson
    print(f"Durbin-Watson (SARIMAX residuals) = {durbin_watson(residuals)}")
    # Shapiro-Wilk
    stat, p_value = shapiro(residuals)
    print(f"Shapiro-Wilk (SARIMAX residuals): stat={stat}, p-value={p_value}")
    # Ljung-Box
    ljung_box_results = acorr_ljungbox(residuals, lags=[10], return_df=True)
    print("Ljung-Box test:\n", ljung_box_results)


print("\n=== Running SARIMAX for each ticker ===")
for ticker in tickers:
    run_sarimax_for_ticker(ticker, all_tickers_df)


# === 7) Exemple d'approche "short-run window" (10 jours avant/après earnings) ===
def build_short_run_df(df, ticker, window_size=10):
    """
    Construit un DataFrame pour chaque fenêtre de 10 jours autour des dates d'earnings.
    """
    subset = df[df['symbol'] == ticker].copy()
    if subset.empty:
        return pd.DataFrame()
    
    # On convertit Date en index pour slicing
    subset.set_index('Date', inplace=True)
    
    short_windows = []
    for d in subset['date'].unique():
        d_start = d - pd.Timedelta(days=window_size)
        d_end = d + pd.Timedelta(days=window_size)
        mask = (subset.index >= d_start) & (subset.index <= d_end)
        wdf = subset.loc[mask].copy()
        
        if not wdf.empty:
            # On ajoute l'earnings_surprise commun
            surprise_val = subset.loc[subset['date'] == d, 'earnings_surprise'].iloc[0]
            wdf['earnings_surprise'] = surprise_val
            short_windows.append(wdf)
            
    if short_windows:
        out = pd.concat(short_windows)
        out.reset_index(inplace=True)
        return out
    else:
        return pd.DataFrame()


for ticker in tickers:
    short_run_data = build_short_run_df(all_tickers_df, ticker, window_size=10)
    if short_run_data.empty:
        print(f"No short-run window for {ticker}.")
        continue
    
    # On tente un petit ARIMA ou SARIMAX sur ce short-run
    y_short = short_run_data['Close']
    X_short = short_run_data[['earnings_surprise']].copy().fillna(0)
    
    # auto_arima sur la période short-run
    arima_model = auto_arima(
        y_short, exogenous=X_short, 
        start_p=1, start_q=1, max_p=3, max_q=3,
        seasonal=False, d=0, 
        trace=False, error_action='ignore', stepwise=True
    )
    print(f"\nShort-run window for {ticker}, auto_arima order: {arima_model.order}")
    
    # Fit statsmodels SARIMAX
    short_sarimax = SARIMAX(y_short, exog=X_short, order=arima_model.order,
                            enforce_stationarity=False, 
                            enforce_invertibility=False).fit(disp=False)
    print(short_sarimax.summary())

    # Plot
    preds_short = short_sarimax.get_prediction(exog=X_short).predicted_mean
    
    plt.figure(figsize=(12,6))
    plt.plot(short_run_data['Date'], y_short, label='Actual', color='blue')
    plt.plot(short_run_data['Date'], preds_short, label='Predicted', color='red')
    plt.title(f'{ticker} - Short-run SARIMAX (±10 jours autour earnings)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

# === 8) Démonstration d'analyse de sentiment via FinBERT & merging ===

print("\n=== Sentiment Analysis (FinBERT) ===")

# 8.1 Prépare le pipeline FinBERT
finbert_model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone", num_labels=3)
finbert_tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
nlp = pipeline("sentiment-analysis", model=finbert_model, tokenizer=finbert_tokenizer)

label_weights = {'Neutral': 0, 'Positive': 1, 'Negative': -1}

def process_text_in_chunks(text, max_chunk_length=512):
    """
    Découpe le texte en chunks <= 512 tokens pour éviter les dépassements de longueur.
    """
    tokenized_text = finbert_tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokenized_text[i:i + max_chunk_length] for i in range(0, len(tokenized_text), max_chunk_length)]
    results = []
    for chunk in chunks:
        chunk_with_special = [finbert_tokenizer.cls_token_id] + chunk + [finbert_tokenizer.sep_token_id]
        if len(chunk_with_special) > 512:
            continue
        chunk_text = finbert_tokenizer.decode(chunk_with_special, skip_special_tokens=True)
        result = nlp(chunk_text)
        results.append(result)
    return results

def calculate_sentiment_score(texts):
    """
    Calcule un score de sentiment moyen en utilisant des poids:
    Positive -> +1, Negative -> -1, Neutral -> 0.
    """
    all_weighted_scores = []
    for txt in texts:
        chunk_results = process_text_in_chunks(txt)
        flat_results = [res for sublist in chunk_results for res in sublist]
        weighted_scores = [label_weights[r['label']] * r['score'] for r in flat_results]
        if weighted_scores:
            avg_score = sum(weighted_scores) / len(weighted_scores)
            all_weighted_scores.append(avg_score)
    if all_weighted_scores:
        return sum(all_weighted_scores) / len(all_weighted_scores)
    else:
        return 0

# Define quarters
symbol_quarters =  {
    'TSLA': [
    "Tesla third quarter 2023 results", "Tesla second quarter 2023 results",
    "Tesla first quarter 2023 results", "Tesla fourth quarter 2022 results",
    "Tesla third quarter 2022 results", "Tesla second quarter 2022 results",
    "Tesla first quarter 2022 results", "Tesla fourth quarter 2021 results",
    "Tesla third quarter 2021 results", "Tesla second quarter 2021 results",
    "Tesla first quarter 2021 results", "Tesla fourth quarter 2020 results", 
    "Tesla third quarter 2020 results", "Tesla second quarter 2020 results", 
    "Tesla first quarter 2020 results", "Tesla fourth quarter 2019 results", 
    "Tesla third quarter 2019 results", "Tesla second quarter 2019 results", 
    "Tesla first quarter 2019 results", "Tesla fourth quarter 2018 results", 
    "Tesla third quarter 2018 results", "Tesla second quarter 2018 results", 
    "Tesla first quarter 2018 results", "Tesla fourth quarter 2017 results", 
    "Tesla third quarter 2017 results", "Tesla second quarter 2017 results", 
    "Tesla first quarter 2017 results", "Tesla fourth quarter 2016 results", 
    "Tesla third quarter 2016 results", "Tesla second quarter 2016 results", 
    "Tesla first quarter 2016 results", "Tesla fourth quarter 2015 results",
    "Tesla third quarter 2015 results", "Tesla second quarter 2015 results", 
    "Tesla first quarter 2015 results", "Tesla fourth quarter 2014 results", 
    "Tesla third quarter 2014 results", "Tesla second quarter 2014 results", 
    "Tesla first quarter 2014 results"
    ],
    
    
        
    'NEE': [
    "NextEra third quarter 2023 results", "NextEra second quarter 2023 results",
    "NextEra first quarter 2023 results", "NextEra fourth quarter 2022 results",
    "NextÈra third quarter 2022 results", "NextEra second quarter 2022 results",
    "NextEra first quarter 2022 results", "NextEra fourth quarter 2021 results",
    "NextEra third quarter 2021 results", "NextEra second quarter 2021 results",
    "NextEra first quarter 2021 results", "NextEra fourth quarter 2020 results", 
    "NextEra third quarter 2020 results", "NextEra second quarter 2020 results", 
    "NextEra first quarter 2020 results", "NextEra fourth quarter 2019 results", 
    "NextEra third quarter 2019 results", "NextEra second quarter 2019 results", 
    "NextEra first quarter 2019 results", "NextEra fourth quarter 2018 results", 
    "NextEra third quarter 2018 results", "NextEra second quarter 2018 results", 
    "NextEra first quarter 2018 results", "NextEra fourth quarter 2017 results", 
    "NextEra third quarter 2017 results", "NextEra second quarter 2017 results", 
    "NextEra first quarter 2017 results", "NextEra fourth quarter 2016 results", 
    "NextEra third quarter 2016 results", "NextEra second quarter 2016 results", 
    "NextEra first quarter 2016 results", "NextEra fourth quarter 2015 results",
    "NextEra third quarter 2015 results", "NextEra second quarter 2015 results", 
    "NextEra first quarter 2015 results", "NextEra fourth quarter 2014 results", 
    "NextEra third quarter 2014 results", "NextEra second quarter 2014 results", 
    "NextEra first quarter 2014 results"
    ],

    'FSLR': [
    "First Solar third quarter 2023 results", "First Solar second quarter 2023 results",
    "First Solar first quarter 2023 results", "First Solar fourth quarter 2022 results",
    "First Solar third quarter 2022 results", "First Solar second quarter 2022 results",
    "First Solar first quarter 2022 results", "First Solar fourth quarter 2021 results", 
    "First Solar third quarter 2021 results", "First Solar second quarter 2021 results",
    "First Solar first quarter 2021 results", "First Solar fourth quarter 2020 results", 
    "First Solar third quarter 2020 results", "First Solar second quarter 2020 results", 
    "First Solar first quarter 2020 results", "First Solar fourth quarter 2019 results", 
    "First Solar third quarter 2019 results", "First Solar second quarter 2019 results", 
    "First Solar first quarter 2019 results", "First Solar fourth quarter 2018 results", 
    "First Solar third quarter 2018 results", "First Solar second quarter 2018 results", 
    "First Solar first quarter 2018 results", "First Solar fourth quarter 2017 results", 
    "First Solar third quarter 2017 results", "First Solar second quarter 2017 results", 
    "First Solar first quarter 2017 results", "First Solar fourth quarter 2016 results", 
    "First Solar third quarter 2016 results", "First Solar second quarter 2016 results", 
    "First Solar first quarter 2016 results", "First Solar fourth quarter 2015 results",
    "First Solar third quarter 2015 results", "First Solar second quarter 2015 results", 
    "First Solar first quarter 2015 results", "First Solar fourth quarter 2014 results", 
    "First Solar third quarter 2014 results", "First Solar second quarter 2014 results", 
    "First Solar first quarter 2014 results"
    ],


    'PLUG': [
    "Plug Power third quarter 2023 results", "Plug Power second quarter 2023 results",
    "Plug Power first quarter 2023 results", "Plug Power fourth quarter 2022 results",
    "Plug Power third quarter 2022 results", "Plug Power second quarter 2022 results",
    "Plug Power first quarter 2022 results", "Plug Power fourth quarter 2021 results", 
    "Plug Power third quarter 2021 results", "Plug Power second quarter 2021 results",
    "Plug Power first quarter 2021 results", "Plug Power fourth quarter 2020 results", 
    "Plug Power third quarter 2020 results", "Plug Power second quarter 2020 results", 
    "Plug Power first quarter 2020 results", "Plug Power fourth quarter 2019 results", 
    "Plug Power third quarter 2019 results", "Plug Power second quarter 2019 results", 
    "Plug Power first quarter 2019 results", "Plug Power fourth quarter 2018 results", 
    "Plug Power third quarter 2018 results", "Plug Power second quarter 2018 results", 
    "Plug Power first quarter 2018 results", "Plug Power fourth quarter 2017 results", 
    "Plug Power third quarter 2017 results", "Plug Power second quarter 2017 results", 
    "Plug Power first quarter 2017 results", "Plug Power fourth quarter 2016 results", 
    "Plug Power third quarter 2016 results", "Plug Power second quarter 2016 results", 
    "Plug Power first quarter 2016 results", "Plug Power fourth quarter 2015 results",
    "Plug Power third quarter 2015 results", "Plug Power second quarter 2015 results", 
    "Plug Power first quarter 2015 results", "Plug Power fourth quarter 2014 results", 
    "Plug Power third quarter 2014 results", "Plug Power second quarter 2014 results", 
    "Plug Power first quarter 2014 results"
    ],


    'GE': [
    "General Electric third quarter 2023 results", "General Electric second quarter 2023 results",
    "General Electric first quarter 2023 results", "General Electric fourth quarter 2022 results",
    "General Electric third quarter 2022 results", "General Electric second quarter 2022 results",
    "General Electric first quarter 2022 results", "General Electric fourth quarter 2021 results", 
    "General Electric third quarter 2021 results", "General Electric second quarter 2021 results",
    "General Electric first quarter 2021 results", "General Electric fourth quarter 2020 results", 
    "General Electric third quarter 2020 results", "General Electric second quarter 2020 results", 
    "General Electric first quarter 2020 results", "General Electric fourth quarter 2019 results", 
    "General Electric third quarter 2019 results", "General Electric second quarter 2019 results", 
    "General Electric first quarter 2019 results", "General Electric fourth quarter 2018 results", 
    "General Electric third quarter 2018 results", "General Electric second quarter 2018 results", 
    "General Electric first quarter 2018 results", "General Electric fourth quarter 2017 results", 
    "General Electric third quarter 2017 results", "General Electric second quarter 2017 results", 
    "General Electric first quarter 2017 results", "General Electric fourth quarter 2016 results", 
    "General Electric third quarter 2016 results", "General Electric second quarter 2016 results", 
    "General Electric first quarter 2016 results", "General Electric fourth quarter 2015 results",
    "General Electric third quarter 2015 results", "General Electric second quarter 2015 results", 
    "General Electric first quarter 2015 results", "General Electric fourth quarter 2014 results", 
    "General Electric third quarter 2014 results", "General Electric second quarter 2014 results", 
    "General Electric first quarter 2014 results"
    ]
}

symbol_quarter_sentiment_scores = {}
for symbol, queries in symbol_quarters.items():
    symbol_quarter_sentiment_scores[symbol] = {}
    for query in queries:
        # On appelle une fonction hypothétique de crawling
        df_crawl = write_crawl_results([query], n=10)  # suppose que cette fonction retourne un df avec colonnes ['body']
        texts = df_crawl['body'].tolist()

        # On calcule un score moyen
        score = calculate_sentiment_score(texts)
        symbol_quarter_sentiment_scores[symbol][query] = score
        print(f"{symbol} / '{query}' => Sentiment Score: {score}")


# 8.2 Convertit ces scores en DataFrame pour potentielle fusion
sentiment_rows = []
for sym, qdict in symbol_quarter_sentiment_scores.items():
    for q, sc in qdict.items():
        sentiment_rows.append((sym, q, sc))

sentiment_df = pd.DataFrame(sentiment_rows, columns=['symbol', 'quarter_query', 'sentiment_score'])
print("\n=== Sentiment DataFrame ===")
print(sentiment_df.head())

# 8.3 Exemple de merge avec all_tickers_df (il faudrait disposer d'une clé "quarter" ou "quarter_query")
# Dans l'exemple suivant, on suppose juste qu'on a 'symbol' et 'quarter_query'. L’utilisateur doit adapter.
# final_merged_df = pd.merge(all_tickers_df, sentiment_df, on=['symbol'], how='left')
# => Adapter la logique de correspondance ticker / quarter.

print("\nScript terminé. Les sections clés ont été montrées (fetch data, stationarity, ARIMA/SARIMAX, sentiment).")
