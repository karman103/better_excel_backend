# CSV Cursor MVP

A simple command-line tool to navigate and analyze CSV files, specifically designed for stock data analysis with AI-powered insights.

## Features

- 📊 **Record Navigation**: Move through CSV records one by one
- 🔍 **Date Search**: Search for specific dates in the data
- 📈 **Statistics**: View summary statistics and performance metrics
- 🎯 **Extreme Values**: Find highest/lowest prices, volumes, and biggest gains/losses
- 📋 **Recent Records**: View a table of recent records around current position
- 🤖 **AI Analysis**: Ask natural language questions about your data

## Versions

### Basic Version (`csv_cursor.py`)
Simple CSV navigation and analysis without AI integration.

### AI-Enhanced Version (`csv_cursor_simple_llm.py`)
Includes intelligent analysis using rule-based AI responses for natural language queries.

### Full MLX Version (`csv_cursor_llm.py`)
Advanced version with MLX integration for more sophisticated LLM analysis (requires MLX installation).

## Usage

### Basic Usage
```bash
python csv_cursor.py
# or
python csv_cursor_simple_llm.py
```

This will load the default Netflix stock data file (`Download Data - STOCK_US_XNAS_NFLX.csv`).

### Custom CSV File
```bash
python csv_cursor_simple_llm.py your_file.csv
```

## Commands

| Command | Description |
|---------|-------------|
| `n` or `next` | Move to next record |
| `p` or `prev` | Move to previous record |
| `g <num>` | Go to specific record number |
| `s <date>` | Search by date (e.g., `s 05/16/2025`) |
| `c` or `current` | Show current record details |
| `r` or `recent` | Show recent records table |
| `sum` | Show summary statistics |
| `e` or `extremes` | Show extreme values |
| `ask <question>` | Ask AI about the data |
| `q` or `quit` | Exit the application |

## AI Analysis Examples

The AI-enhanced version can answer questions like:

- **"What is the trend?"** - Analyzes moving averages and trend direction
- **"How is the volume today?"** - Compares current volume to average
- **"What is the overall performance?"** - Shows total return over the period
- **"How volatile is the stock?"** - Analyzes daily price ranges
- **"What are the support and resistance levels?"** - Shows price boundaries
- **"What are the patterns?"** - Analyzes consecutive up/down days

## Example Session

```
🎯 CSV Cursor with AI Analysis - Download Data - STOCK_US_XNAS_NFLX.csv
==================================================
✅ Loaded 250 records from Download Data - STOCK_US_XNAS_NFLX.csv

[1/250] > s 05/16/2025

📊 Record 34 of 250
==================================================
Date:     05/16/2025
Open:     $1193.14
High:     $1196.50
Low:      $1179.39
Close:    $1191.53
Volume:   4,698,353
Change:   📉 $-1.61 (-0.13%)

[34/250] > ask what is the trend

🤖 AI Analysis for: 'what is the trend'
============================================================
Based on the moving averages, the current trend appears to be Bearish.
============================================================

[34/250] > ask how volatile is the stock

🤖 AI Analysis for: 'how volatile is the stock'
============================================================
The average daily price range is 2.62%, indicating low volatility.
============================================================
```

## Requirements

### Basic Version
- Python 3.6+
- No external dependencies (uses only standard library)

### AI-Enhanced Version
- Python 3.6+
- pandas
- numpy

### Full MLX Version
- Python 3.6+
- mlx
- transformers
- pandas
- numpy

Install dependencies:
```bash
pip install -r requirements.txt
```

## File Format

The tool expects CSV files with the following columns:
- Date
- Open
- High
- Low
- Close
- Volume

The Netflix stock data file is already in the correct format.

## Quick Start

1. Make sure you have Python installed
2. Install dependencies: `pip install pandas numpy`
3. Run the AI-enhanced version: `python csv_cursor_simple_llm.py`
4. Use the commands to navigate and analyze your data
5. Try asking questions like:
   - `ask what is the trend`
   - `ask how is the volume today`
   - `ask what is the overall performance`
   - `ask how volatile is the stock`

## AI Capabilities

The AI analyzer can understand and respond to queries about:

- **Trend Analysis**: Moving averages, trend direction, technical indicators
- **Volume Analysis**: Trading activity, volume patterns, average comparisons
- **Performance Metrics**: Total returns, gains/losses, period performance
- **Volatility Analysis**: Price ranges, volatility levels, daily swings
- **Price Action**: Daily moves, intraday patterns, price behavior
- **Pattern Recognition**: Consecutive days, streaks, recurring patterns
- **Support/Resistance**: Price boundaries, high/low levels

## Advanced Features

- **Pattern Detection**: Automatically identifies up/down streaks and volatile days
- **Trend Strength**: Calculates trend strength using moving average divergence
- **Volume Analysis**: Compares current volume to historical averages
- **Risk Assessment**: Evaluates volatility and price range patterns
- **Performance Tracking**: Monitors total returns and period performance 