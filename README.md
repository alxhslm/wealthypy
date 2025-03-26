# :chart: Wealthypy

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://wealthypy.streamlit.app/)

## Overview
This is a **Streamlit-based portfolio simulation tool** that helps visualize long-term investment growth using a rational investing approach, as outlined in *Investing Demystified* by Lars Kroijer. The tool allows users to model their retirement savings (e.g., SIPP) using a mix of global equity index funds and government bonds, accounting for inflation, returns, and volatility.

## Features
- **Customizable Inputs:**
  - Starting portfolio amount
  - Annual inflation rate
  - Variable monthly contributions and allocation
- **Monte Carlo simulations** assuming each asset has a fixed average return and constant volality.
- **Backtesting** using historic returns from Yahoo! finance
- **Interactive visualization** of portfolio growth over time using Plotly.

## Installation
Ensure you have Python installed, then install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the Streamlit app with:
```bash
streamlit run app.py
```
Then, open the provided URL in your browser and adjust the parameters to explore different investment scenarios.

## License
This project is open-source and follows the MIT License.

## Acknowledgments
Inspired by the principles from *Investing Demystified* by Lars Kroijer.

