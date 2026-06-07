<div align="center">

# Forecasting Renewable Energy Stock Returns with Earnings-Call Sentiment

### A FinBERT × SARIMAX framework reveals a **statistically significant regime shift** in sentiment-return dynamics across the COVID-19 break

[![Python](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FinBERT](https://img.shields.io/badge/NLP-FinBERT-FF6B35)](https://huggingface.co/yiyanghkust/finbert-tone)
[![Statsmodels](https://img.shields.io/badge/SARIMAX-statsmodels-4B8BBE)](https://www.statsmodels.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Thesis](https://img.shields.io/badge/Columbia%20University-QMSS%202024-9C002B)](https://qmss.columbia.edu/)
[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Raeus1901/finbert-sarimax-energy-forecasting/main?labpath=notebooks%2FMaster_Thesis.ipynb)

</div>

---

## 🎯 Key Finding

> **The sign of the sentiment-return coefficient inverted from +79.55 (pre-COVID, p<0.001) to −90.23 (post-COVID, p=0.039) for the renewable energy universe — a complete reversal of investor reaction to corporate communication.**

![Sentiment-return inversion plot](assets/hero_finding.png)

**Interpretation.** Before March 2020, positive earnings-call sentiment predicted positive forward returns (consistent with the *information content hypothesis*, Tetlock 2007). Post-COVID, the relationship inverted: positive sentiment now precedes underperformance, suggesting either (i) **structural skepticism** toward management's optimism in a post-crisis regime, or (ii) **contrarian flows** from informed investors fading sell-side enthusiasm. This regime shift is robust to standard SARIMAX diagnostics (Ljung-Box Q, Jarque-Bera, ADF/KPSS).

---

## 📊 At a Glance

| Component | Detail |
|---|---|
| **Universe** | 5 renewable-energy firms (FSLR, GE, NEE, TSLA, PLUG), 2014–2023 |
| **Frequency** | Quarterly observations (~N=38 per firm; pooled N=114 pre-COVID, N=77 post-COVID) |
| **Sentiment engine** | FinBERT (Huang, Wang & Yang 2023) on 2,000+ scraped earnings articles |
| **Forecasting model** | SARIMAX with exogenous sentiment regressor |
| **Diagnostics** | Ljung-Box, Jarque-Bera, ADF, KPSS, residual normality |
| **Key result** | Sentiment coefficient sign inversion at COVID-19 structural break |

---

## 🚀 Quick Start

```bash
# Clone & install
git clone https://github.com/Raeus1901/finbert-sarimax-energy-forecasting.git
cd finbert-sarimax-energy-forecasting
pip install -r requirements.txt

# Reproduce the hero plot
python notebooks/generate_hero_plot.py

# Run the full thesis pipeline (notebook)
jupyter lab notebooks/Master_Thesis.ipynb
```

**Or click the Binder badge above to run everything in the browser — no install required.**

---

## 📚 Methodology

The pipeline integrates three components:

1. **Sentiment extraction** — FinBERT (finance-domain-tuned BERT) processes 2,000+ earnings-related articles per firm-quarter, returning probability-weighted sentiment scores.
2. **Time-series modeling** — SARIMAX(p, d, q)(P, D, Q, s) with sentiment as exogenous regressor, orders selected via AIC + residual diagnostics.
3. **Structural break analysis** — Pre/post-COVID samples fit independently to test coefficient stability across the regime.

Full derivation, citations, and reproducibility notes in [`notebooks/Master_Thesis.ipynb`](notebooks/Master_Thesis.ipynb).

---

## ⚠️ Limitations & Robustness

This thesis is presented with explicit acknowledgement of its statistical constraints:

- **Sample size.** N=114 pooled observations pre-COVID (5 firms × ~23 quarters), N=77 post-COVID (5 firms × ~16 quarters) is below conventional thresholds for stable SARIMAX inference [Hyndman & Athanasopoulos 2021, §9].
- **Multiple testing.** 5 tickers × multiple sentiment specifications without Bonferroni/FDR correction.
- **Bootstrap stability.** Coefficients warrant validation via block bootstrap [Politis & Romano 1994] before causal interpretation.
- **Structural break detection.** COVID split is exogenously imposed; Bai-Perron [2003] would identify breakpoints endogenously.
- **Crawler fragility.** Article scraping via Google News URLs is non-deterministic across runs.

See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) for the full discussion.

---

## 📖 References

- Huang, A. H., Wang, H., & Yang, Y. (2023). *FinBERT: A large language model for extracting information from financial text.* Contemporary Accounting Research, 40(2).
- Loughran, T., & McDonald, B. (2011). *When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks.* The Journal of Finance, 66(1).
- Tetlock, P. C. (2007). *Giving content to investor sentiment: The role of media in the stock market.* The Journal of Finance, 62(3).
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.

Full bibliography in [`docs/references.bib`](docs/references.bib).

---

## 👤 Author

**Jean Treves** — M.A. Quantitative Methods in the Social Sciences, Columbia University (GPA 3.92, 2024)
[LinkedIn](https://www.linkedin.com/in/jean-treves-bbaa91257/) • [GitHub](https://github.com/Raeus1901) • jdt2175@columbia.edu

---

*If you find this work useful for your research, please cite the thesis (BibTeX in `docs/references.bib`).*
