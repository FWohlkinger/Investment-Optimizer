# 📈 Investment Optimizer
The **Investment Optimizer** is a Python-based tool that simulates the growth of an investment portfolio over time, accounting for various financial parameters such as deposit fees, management fees, dynamic savings increases, and both fixed and volatile returns. The tool is designed for individuals interested in financial modeling, data analysis, and investment strategy development.

It supports two main calculation modes:

- **Fixed Return**: Assumes a constant rate of return throughout the investment period, ideal for modeling savings accounts, bonds, or other fixed-income investments.
- **Market Simulation**: Simulates the investment in a volatile market (e.g., a stock ETF), where the return rate is subject to market fluctuations and volatility.

The tool calculates the final portfolio value, compares it with a benchmark (zero-cost, maximum-performance scenario), and visualizes the results with clear, informative plots.

## 📂 Table of Contents
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Parameters](#-parameters)
    - [Fixed Return Mode](#-fixed-return-mode)
    - [Market Simulation Mode](#-market-simulation-mode)
- [Results and Visualization](#-results-and-visualization)
- [Comparing Investment Scenarios](#-comparing-investment-scenarios)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

## 🌟 Features
- **Supports both fixed and volatile returns**: Simulate traditional fixed-income investments or market-driven assets.
- **Customizable fees**: Model deposit fees, management fees, and closing costs, including both percentage-based and fixed amounts.
- **Dynamic savings rates**: Include annual increases in monthly savings rates (e.g., salary increases).
- **Benchmark comparison**: Calculate performance relative to a zero-cost, maximum-performance benchmark scenario.
- **Comprehensive reporting**: Display detailed investment data in tabular form and visualize the performance over time.
- **Cost breakdown**: Differentiate between direct and indirect costs and visualize their impact on final returns.
- **Seed-based simulation**: Reproduce random results using a seed for random number generation (for volatile returns).

## 🛠 Installation
### Prerequisites
- Python 3.7+
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Tabulate](https://pypi.org/project/tabulate/)

### Installation Steps
1. Clone the repository:
```bash
git clone https://github.com/FWohlkinger/Investment-Optimizer.git
```

2. Navigate to the project directory:
```
cd investment-optimizer
```

3. Install the required Python packages:
```bash
pip install -r requirements.txt
```

## ▶️ Usage
After installation, run the latest version of the **Investment Optimizer** to start the calculation.
```
python investment_optimizer_v3.0.0_de.py
```

The program will guide you through setting up your investment parameters. You can use default parameters or customize them based on your scenario.

## ⚙️ Parameters
The **Investment Optimizer** uses a set of parameters that define the investment scenario. You can choose between two different modes: **Fixed Return Mode** and **Market Simulation Mode**.

### Fixed Return Mode
This mode assumes a constant annual return rate throughout the investment period, making it ideal for simulating fixed-income instruments like bonds, savings accounts, or other guaranteed return products.

**Key Parameters**:
- **Initial Investment**: Starting amount of the investment.
- **Duration**: Investment period (in years).
- **Monthly Savings**: Initial amount added monthly to the portfolio.
- **Dynamic Savings Increase**: Annual increase in the monthly savings rate (percentage).
- **Yield Interval**: Determines how often returns are distributed (monthly, quarterly, annually).
- **Fees**: Including deposit fees, annual management fees, and closing costs.

### Market Simulation Mode
This mode simulates an investment exposed to market volatility, such as stocks, ETFs, or mutual funds. It uses an autoregressive process to model monthly return rates based on previous performance and market volatility.

**Key Parameters**:
- **Initial Investment**: Starting amount of the investment.
- **Duration**: Investment period (in years).
- **Monthly Savings**: Initial amount added monthly to the portfolio.
- **Dynamic Savings Increase**: Annual increase in the monthly savings rate (percentage).
- **Expected Return Rate**: Average annual return rate (percentage).
- **Market Volatility**: Volatility of returns (percentage) – a higher value indicates a more volatile investment.
- **Fees**: Including deposit fees, annual management fees, and closing costs.
- **Seed**: Optional value to reproduce the simulation's random results.

## 📊 Results and Visualization
After completing the simulation, the tool generates a detailed report of the investment's performance, including:
- **Month-by-month balance**: See how your portfolio grows over time.
- **Total returns**: Compare your actual returns to a zero-cost benchmark.
- **Direct and indirect costs**: Understand how fees and missed opportunities impact your final portfolio value.

### Visualization
The tool automatically generates several plots to help you analyze the results:
- **Asset Growth**: Track the development of your portfolio over time.
- **Cumulative Deposits**: Compare how much you’ve invested vs. your portfolio’s growth.
- **Total Costs**: Visualize the direct and indirect costs that have reduced your returns.
- **Benchmark Comparison**: See how your portfolio compares to a zero-cost, maximum-performance scenario.

## 📈 Comparing Investment Scenarios
The Investment Optimizer allows you to run multiple simulations and compare different scenarios. For example:
- **Scenario 1**: Simulate a conservative bond investment with a fixed 3% return and low fees.
- **Scenario 2**: Simulate a volatile stock ETF investment with a 7% expected return and higher volatility.

At the end of each run, the tool compares the parameters and results of different investment models, providing insights such as:
- Differences in final portfolio value.
- Impact of fees on performance.
- Return on Investment (ROI) before and after taxes.
This comparison functionality is ideal for testing different strategies and selecting the best option for long-term growth.

## 🚀 Roadmap
This roadmap outlines the planned features and improvements for the Investment Optimizer, organized by development effort and logical sequence. Some features depend on others, and are thus planned to be implemented in a stepwise fashion.

### 1. Fixed Management Fees
- **Description**: Add support for *fixed management fees*, in addition to the existing percentage-based fees.
- **Purpose**: Enable more detailed cost modeling, akin to the way deposit fees and closing costs are handled.

### 2. Flexible Additional Deposits
- **Description**: Allow users to *schedule additional, one-time or recurring deposits* (at any chosen time, amount, or frequency), subject to the same fee structure as regular contributions.
- **Purpose**: Provide flexibility for modeling varying investment contributions, a feature common in real-world investment strategies.

### 3. Pause Dynamic Savings Increase
- **Description**: Introduce an option to *pause the dynamic annual increase in monthly savings* at a user-specified time, while allowing the investment to continue under all other conditions.
- **Purpose**: Simulate realistic financial situations where savings growth may plateau, but the investment strategy remains active.

### 4. Contribution Holiday (No More Savings)
- **Description**: Allow users to *stop making further contributions* after a user-defined point, while continuing the investment until its end. Ongoing management fees and closing costs will still apply.
- **Purpose**: Simulate periods where contributions are stopped (e.g., due to financial constraints), while the investment strategy remains active.

### 5. Monte-Carlo Simulation
- **Description**: Implement *Monte-Carlo simulation* to model the potential outcomes of volatile markets (such as stock ETFs) based on random variations in return rates. This allows for a more realistic risk and return analysis.
- **Purpose**: Provide stochastic simulations for investments that rely on market conditions, enhancing the accuracy of future projections.

### 6. GARCH Model for Volatility Simulation
- **Description**: Introduce the **GARCH model** (*Generalized Autoregressive Conditional Heteroskedasticity*) to simulate volatility dynamics, improving the realism of returns in the Monte-Carlo simulation.
- **Purpose**: Provide a more accurate simulation of asset price volatility, particularly for investments like ETFs or stocks.

### 7. Backtesting Mode
- **Description**: Allow users to apply investment parameters to historical stock or ETF data and assess performance under real-world conditions.
- **Purpose**: Validate and optimize investment strategies by testing them against real historical data.

### 8. Machine Learning-Based Prediction Model (Backtesting & Forecasting Integration)
- **Description**: Develop a *Machine-Learning-based model* (*Random Forest* or *Gradient Boosting*) *to predict future market performance based on historical data*. Integrate this with a Backtesting mode that extracts market parameters (volatility, seasonal trends, etc.) and uses them for future projections.
- **Purpose**: Use historical data to generate insights for future predictions, combining the realism of past market trends with advanced forecasting techniques.

### 9. Currency Selection
- **Description**: Add *support for multiple currencies* (**€**, **$**, **£**).
- **Purpose**: Extend usability to international users and facilitate investments across multiple regions.

### 10. Advanced Visualization & Result Analysis
- **Description**: Add more *comprehensive visualizations* to help users analyze their investment performance:
    - **Pie charts** for cost breakdowns (deposit fees, management fees, closing costs).
    - **Bar charts** for comparing key parameters between different scenarios (e.g., total deposits, final portfolio value, total costs).
- **Purpose**: Enhance the clarity and depth of result interpretation through intuitive visual representations.

### 11. Backtesting and Prediction Expansion with Machine Learning
- **Description**: Further enhance the machine-learning prediction model by integrating *sophisticated time series analysis* using methods like **LSTM** (*Long Short-Term Memory*) networks, improving the model's ability to forecast long-term trends and cyclical behavior.
- **Purpose**: Expand the depth and accuracy of the forecast model by incorporating advanced temporal data patterns.

## 🤝 Contributing
Contributions are welcome! If you find a bug or have a suggestion, feel free to open an issue or submit a pull request. Please ensure that your contributions align with the project’s goals and coding style.

## 📜 License
This project is licensed under the MIT License – see the [LICENSE](https://github.com/FWohlkinger/Investment-Optimizer/blob/main/LICENSE) file for details.
