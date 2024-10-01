##################################
###    Investment Optimizer    ###
##################################
### Author: Florian Wohlkinger ###
### Date: 01.10.2024           ###
### Version: 3.0.0             ###
##################################

# Changes in this version:
#   - Implementation of date functionality (adds date column to dataframe, calculates end date etc.)
#   - Revision of terms and parameter names
#   - Implementation of deposit fees for the initial investment sum
#   - Implementation of new cost parameter "annual management fee" (% of balance)
#   - Calculation of maximal return Benchmark (zero-cost scenario)
#   - Calculation of "indirect costs" (missed returns resulting from costs)
#   - Added docstrings for all classes and methods

# ==================================================================================================================== #

# Import modules
import pandas as pd                                 # for Data Frame
import numpy as np                                  # for normally distributed random variable
import matplotlib.pyplot as plt                     # for visualization
from matplotlib.ticker import MultipleLocator       # for visualization (axis labeling)
import json                                         # for saving the investment parameters
from tabulate import tabulate                       # for scenario comparison
import calendar                                     # for handling month-end dates
from datetime import datetime, date, timedelta      # for date-based calculations
from dateutil.relativedelta import relativedelta    # for date-based calculations incl. gap years

# ----------------------------------------------- Save & Load Parameters -----------------------------------------------

def load_parameters(file_name):
    """
    Loads investment parameters from a JSON file.

    Parameters:
        file_name (str): The name of the file to load parameters from

    Returns:
        dict or None: A dictionary of parameters if successful, None if the file is not found
    """
    try:
        with open(file_name, 'r') as f:
            params = json.load(f)
        if 'start_date' in params:
            params['start_date'] = datetime.strptime(params['start_date'], '%d.%m.%Y').date()
        return params
    except FileNotFoundError:
        return None

def save_parameters(parameters, file_name):
    """
    Saves investment parameters to a JSON file.

    Parameters:
        parameters (InvestmentParameters): The investment parameters to save
        file_name (str): The name of the file to save parameters to
    """
    params_dict = vars(parameters)
    params_dict['initial_monthly_savings'] = parameters.initial_monthly_savings
    params_dict['start_date'] = parameters.start_date.strftime('%d.%m.%Y')  # convert date to string
    with open(file_name, 'w') as f:
        json.dump(params_dict, f, indent=2)

# Define filenames
DEFAULT_PARAMS_FILE = 'default_params.json'
USER_PARAMS_FILE = 'user_params.json'

# ------------------------------------------------ InvestmentParameters ------------------------------------------------

class InvestmentParameters:
    def __init__(self, initial_investment, duration, monthly_savings, dynamic_increase, return_rate,
                 market_volatility, yield_interval, deposit_fee_percentage, deposit_fee_fixed,
                 annual_management_fee_percentage, fee_payment_option, closing_costs_percentage, closing_costs_fixed, seed=None, start_date=None):
        """
        Initializes the investment parameters.

        Parameters:
            initial_investment (float): The initial investment amount
            duration (int): The investment duration in years
            monthly_savings (float): The initial monthly savings
            dynamic_increase (float): The annual increase in savings rate as a percentage
            return_rate (float): The expected annual return rate as a percentage
            market_volatility (float): The market volatility as a percentage
            yield_interval (int): The interval for yield distribution (1: annually, 2: quarterly, 3: monthly)
            deposit_fee_percentage (float): The percentage fee deducted from deposits
            deposit_fee_fixed (float): The fixed fee deducted from deposits
            annual_management_fee_percentage (float): The annual management fee as a percentage of the investment
            fee_payment_option (int): The option for fee payment (1: monthly, 2: annually)
            closing_costs_percentage (float): The closing costs as a percentage of the final assets
            closing_costs_fixed (float): The fixed closing costs deducted from final assets
            seed (int, optional): The seed value for random number generation
            start_date (date, optional): The start date of the investment
        """
        self.initial_investment = initial_investment
        self.duration = duration
        self.initial_monthly_savings = monthly_savings
        self.monthly_savings = monthly_savings
        self.dynamic_increase = dynamic_increase
        self.return_rate = return_rate
        self.market_volatility = market_volatility
        self.yield_interval = yield_interval
        self.deposit_fee_percentage = deposit_fee_percentage            # Percentage deduction from deposits
        self.deposit_fee_fixed = deposit_fee_fixed                      # Fixed amount --> deduction from monthly deposits
        self.annual_management_fee_percentage = annual_management_fee_percentage  # Jährliche Verwaltungskosten in Prozent
        self.fee_payment_option = fee_payment_option                    # 1 für monatlich, 2 für jährlich
        self.closing_costs_percentage = closing_costs_percentage        # Percentage deduction from final assets (before taxes)
        self.closing_costs_fixed = closing_costs_fixed                  # Fixed amount --> deduction from final assets (before taxes)
        self.seed = int(seed) if seed is not None else None
        self.start_date = start_date or datetime.now().date()

# ------------------------------------------------ InvestmentCalculator ------------------------------------------------

class InvestmentCalculator:
    """
    Initializes the investment calculator with the given parameters.

    Parameters:
        parameters (InvestmentParameters): The investment parameters for calculations
    """
    def __init__(self, parameters):
        self.parameters = parameters
        self.data = []
        self.total_deposits = parameters.initial_investment
        self.total_return = 0
        self.total_benchmark_return = 0
        self.total_deposit_fees = 0
        self.total_management_fees = 0
        self.total_fees = 0
        self.original_start_date = parameters.start_date
        # Set seed value
        if self.parameters.seed is not None:
            np.random.seed(self.parameters.seed)
        self.end_date = self._calculate_end_date()

    def _calculate_end_date(self):
        """
        Calculates the end date of the investment based on the start date and duration.

        Returns:
            date: The calculated end date
        """
        end_date = self.parameters.start_date + relativedelta(years=self.parameters.duration, days=-1)
        return end_date

    def _generate_date_column(self):
        """
        Generates a list of dates for each month of the investment duration.

        Returns:
            list: A list of formatted date strings
        """
        dates = [(self.parameters.start_date - timedelta(days=1)).strftime('%d.%m.%Y')]  # Row 0

        current_date = self.parameters.start_date
        while len(dates) < len(self.data) - 1:
            dates.append(current_date.strftime('%d.%m.%Y'))
            current_date = self._add_one_month(current_date)

        dates.append(self.end_date.strftime('%d.%m.%Y'))  # Last row
        return dates

    def _add_one_month(self, date):
        """
        Adds one month to the given date, accounting for month-end dates.

        Parameters:
            date (date): The original date to which one month will be added

        Returns:
            date: The new date after adding one month
        """
        original_day = self.original_start_date.day
        next_month = date.replace(day=1) + timedelta(days=32)
        next_month = next_month.replace(day=1)
        last_day_of_next_month = calendar.monthrange(next_month.year, next_month.month)[1]

        if original_day > last_day_of_next_month:
            return next_month.replace(day=last_day_of_next_month)
        else:
            return next_month.replace(day=original_day)

    def calculate(self):
        """
        Performs the investment calculations and returns a DataFrame with the results.

        Returns:
            DataFrame: A DataFrame containing the investment data over time
        """
        self.data = []
        balance = self.parameters.initial_investment
        benchmark_balance = self.parameters.initial_investment
        monthly_savings = self.parameters.initial_monthly_savings
        cumulative_direct_costs = 0

        # Calculate deposit fees on the initial investment
        initial_deposit_fee_percentage = self.parameters.initial_investment * self.parameters.deposit_fee_percentage / 100
        initial_deposit_fees = initial_deposit_fee_percentage + self.parameters.deposit_fee_fixed
        balance -= initial_deposit_fees  # Deduct fees from the initial investment
        self.total_deposit_fees += initial_deposit_fees  # Add the initial deposit fee to cumulative deposit fees
        cumulative_direct_costs += initial_deposit_fees  # Add the initial deposit fee to cumulative direct costs

        # Row 0 before start of duration ("Month 0") --> Duration starts with initial investment
        self.data.append([0, 0, 0, 0, self.parameters.initial_investment, initial_deposit_fees,
                          self.parameters.initial_investment - initial_deposit_fees, 0,
                          self.parameters.initial_investment - initial_deposit_fees, 0, 0, cumulative_direct_costs,
                          benchmark_balance, 0, 0])

        # Loop for calendar year
        for year in range(1, self.parameters.duration + 1):
            annual_return = 0
            benchmark_annual_return = 0
            previous_month_return_rate = self.parameters.return_rate

            # Loop for individual months
            for month in range(1, 13):
                # The return rate for the respective row
                monthly_return_rate = self._calculate_return_rate(previous_month_return_rate)
                previous_month_return_rate = monthly_return_rate  # for AR-model

                # Month Start: Carry over the balance from the previous row
                month_start = balance
                benchmark_month_start = benchmark_balance

                # Calculate deposit fees
                deposit_fees, investment_rate = self._calculate_deposit_fees(monthly_savings)

                # Add monthly savings (with/without cost deduction) to the balance
                balance += investment_rate
                benchmark_balance += monthly_savings

                # Calculate capital gains
                monthly_return = round(balance * ((monthly_return_rate / 100) / 12), 2)
                monthly_benchmark_return = round(benchmark_balance * ((monthly_return_rate / 100) / 12), 2)

                # Calculate annual management fees
                fee_to_record = self._calculate_management_fee(balance, month)
                self.total_management_fees += fee_to_record

                # Update stored parameters
                self._update_total_values(monthly_return, monthly_benchmark_return, deposit_fees, monthly_savings, fee_to_record)

                # Update cumulative total costs
                cumulative_direct_costs += deposit_fees + fee_to_record

                # Extend DataFrame with the current row
                self.data.append([year, month, monthly_return_rate, round(month_start, 2),
                                  monthly_savings, deposit_fees, investment_rate, round(monthly_return, 2),
                                  round(balance, 2), round(fee_to_record, 2), 0, round(cumulative_direct_costs, 2),
                                  benchmark_balance, round(monthly_benchmark_return, 2), 0])

                # Add returns to the balance
                balance, annual_return = self._process_yield_distribution(balance, annual_return, monthly_return,month)
                benchmark_balance, benchmark_annual_return = self._process_benchmark_yield_distribution(
                    benchmark_balance, benchmark_annual_return, monthly_benchmark_return, month)

                # Deduct management fees from the balance
                balance -= fee_to_record

            monthly_savings = self._update_savings_rate(year, monthly_savings)

        self._add_closing_data(balance, benchmark_balance)

        df = pd.DataFrame(self.data, columns=["Year", "Month", "Return Rate", "Month Start", "Monthly Savings",
                                              "Fees", "Monthly Investment Rate", "Monthly Return", "Month End",
                                              "Costs", "Yield Distribution", "Total Direct Costs", "Benchmark Balance",
                                              "Monthly Benchmark Return", "Benchmark Yield Distribution"])
        df.insert(2, "Date", self._generate_date_column())
        return df

    def _calculate_return_rate(self, previous_month_return_rate):
        """
        Calculates the monthly return rate based on the previous month's return rate.

        Parameters:
            previous_month_return_rate (float): The return rate from the previous month

        Returns:
            float: The calculated monthly return rate
        """
        ### Procedure for "volatile return rates": Generate a normally distributed random variable
        #   with dependency on the previous month's return rate --> autoregressive component (AR)
        return_stddev = self.parameters.return_rate * self.parameters.market_volatility / 100
        random_variable = np.random.normal(0, return_stddev)
        AR_parameter = 0.5  # AR parameter for the AR(1) process

        if return_stddev == 0:  # Fixed return rate (no volatility)
            return self.parameters.return_rate
        else:  # Variable return rate --> AR model
            return self.parameters.return_rate + AR_parameter * (
                    previous_month_return_rate - self.parameters.return_rate) + random_variable

    def _update_savings_rate(self, year, current_savings):
        """
        Updates the savings rate based on the dynamic increase percentage.

        Parameters:
            year (int): The current year of the investment
            current_savings (float): The current savings amount

        Returns:
            float: The updated savings rate
        """
        if self.parameters.dynamic_increase > 0 and year < self.parameters.duration:
            dynamic_amount = round(current_savings * (self.parameters.dynamic_increase / 100), 2)
            return round(current_savings + dynamic_amount, 2)
        return current_savings

    def _calculate_deposit_fees(self, monthly_savings):
        """
        Calculates the deposit fees based on the monthly savings.

        Parameters:
            monthly_savings (float): The amount of monthly savings

        Returns:
            tuple: A tuple containing the deposit fees and the investment rate after fees
        """
        deposit_fee_percentage = round(monthly_savings * self.parameters.deposit_fee_percentage / 100, 2)
        deposit_fees = deposit_fee_percentage + self.parameters.deposit_fee_fixed
        investment_rate = monthly_savings - deposit_fees
        return deposit_fees, investment_rate

    def _update_total_values(self, monthly_return, monthly_benchmark_return, deposit_fees, monthly_savings, fee_to_record):
        """
        Updates the total values for returns, deposits, and fees.

        Parameters:
            monthly_return (float): The return for the current month
            monthly_benchmark_return (float): The benchmark return for the current month
            deposit_fees (float): The fees deducted from deposits
            monthly_savings (float): The amount of monthly savings
            fee_to_record (float): The management fee for the current month
        """
        self.total_return += monthly_return
        self.total_benchmark_return += monthly_benchmark_return
        self.total_deposits += monthly_savings
        self.total_deposit_fees += deposit_fees
        self.total_fees += fee_to_record

    def _calculate_management_fee(self, balance, month):
        """
        Calculates the management fee based on the balance and the month.

        Parameters:
            balance (float): The current balance of the investment
            month (int): The current month of the investment

        Returns:
            float: The calculated management fee
        """
        if self.parameters.fee_payment_option == 1:  # monthly
            return round((balance * self.parameters.annual_management_fee_percentage / 100) / 12, 2)
        elif self.parameters.fee_payment_option == 2 and month == 12:  # annually
            return round(balance * self.parameters.annual_management_fee_percentage / 100, 2)
        else:
            return 0

    def _process_yield_distribution(self, balance, annual_return, monthly_return, month):
        """
        Processes the yield distribution based on the investment strategy.

        Parameters:
            balance (float): The current balance of the investment
            annual_return (float): The accumulated annual return
            monthly_return (float): The return for the current month
            month (int): The current month of the investment

        Returns:
            tuple: Updated balance and annual return after yield distribution
        """
        annual_return += monthly_return
        if self.parameters.yield_interval == 1 and month == 12:  # annually
            self.data[-1][-5] = round(annual_return, 2)
            balance += self.data[-1][-5]
            annual_return = 0
        elif self.parameters.yield_interval == 2 and month % 3 == 0:  # quarterly
            self.data[-1][-5] = round(annual_return, 2)
            balance += self.data[-1][-5]
            annual_return = 0
        elif self.parameters.yield_interval == 3:  # monthly
            self.data[-1][-5] = round(monthly_return, 2)
            balance += self.data[-1][-5]
        return balance, annual_return

    def _process_benchmark_yield_distribution(self, benchmark_balance, benchmark_annual_return,
                                              monthly_benchmark_return, month):
        """
        Processes the yield distribution for the benchmark investment.

        Parameters
            benchmark_balance (float): The current balance of the benchmark investment
            benchmark_annual_return (float): The accumulated annual return for the benchmark
            monthly_benchmark_return (float): The return for the current month for the benchmark
            month (int): The current month of the investment

        Returns:
            tuple: Updated benchmark balance and annual return after yield distribution
        """
        benchmark_annual_return += monthly_benchmark_return
        if self.parameters.yield_interval == 1 and month == 12:  # annually
            self.data[-1][-1] = round(benchmark_annual_return, 2)
            benchmark_balance += self.data[-1][-1]
            benchmark_annual_return = 0
        elif self.parameters.yield_interval == 2 and month % 3 == 0:  # quarterly
            self.data[-1][-1] = round(benchmark_annual_return, 2)
            benchmark_balance += self.data[-1][-1]
            benchmark_annual_return = 0
        elif self.parameters.yield_interval == 3:  # monthly
            self.data[-1][-1] = round(monthly_benchmark_return, 2)
            benchmark_balance += self.data[-1][-1]
        return benchmark_balance, benchmark_annual_return

    def _add_closing_data(self, balance, benchmark_balance):
        """
        Adds closing data to the investment data set.

        Parameters:
            balance (float): The final balance of the investment
            benchmark_balance (float): The final balance of the benchmark investment
        """
        final_assets = balance
        final_benchmark_assets = benchmark_balance
        closing_costs_variable = round(final_assets * self.parameters.closing_costs_percentage / 100, 2)
        closing_costs = closing_costs_variable + self.parameters.closing_costs_fixed
        final_assets_after_closing_costs = final_assets - closing_costs
        total_deposits = 0

        self.data.append([self.parameters.duration + 1, 0, 0, final_assets , total_deposits, closing_costs, 0, 0,
                          final_assets_after_closing_costs, self.total_fees, self.total_return,
                          self.total_deposit_fees + closing_costs + self.total_fees,
                          final_benchmark_assets, 0, self.total_benchmark_return])


# --------------------------------------------------- Visualization ---------------------------------------------------

class InvestmentVisualizer:
    def __init__(self, df, parameters, calculator):
        """
        Initializes the investment visualizer.

        Parameters:
            df (DataFrame): The DataFrame containing investment data
            parameters (InvestmentParameters): The investment parameters used for visualization
            calculator (InvestmentCalculator): The investment calculator used for calculations
        """
        self.df = df
        self.df2 = df.copy()
        self.parameters = parameters
        self.calculator = calculator

    def visualize(self):
        """
        Visualizes the investment data using plots.

        Returns:
            Figure: The generated figure for the investment visualization
        """
        self.df = self.df.iloc[1:].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(10, 6))

        # Call plots
        self._plot_asset_development(ax)
        self._plot_deposits(ax)
        self._plot_costs(ax)
        self._plot_benchmark(ax)

        # Set axes and title
        self._set_axes_and_title(ax)
        self._add_legend(ax)

        plt.legend(loc='upper center')  # Alt: 'lower right' | 'center right'
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return fig

    def _plot_asset_development(self, ax):
        """
        Plots the development of the asset over time.

        Parameters:
            ax (Axes): The matplotlib axes to plot on
        """
        ax.step(self.df.index, self.df['Month End'], where='post', label='Vermögensstand', linestyle='-')
        ax.fill_between(self.df.index, self.df['Month End'], step='post', color='blue', alpha=0.3)

    def _plot_deposits(self, ax):
        """
        Plots the cumulative deposits over time.

        Parameters:
            ax (Axes): The matplotlib axes to plot on
        """
        cumulative_deposits = self.df['Monthly Savings'].cumsum() + self.parameters.initial_investment
        ax.step(self.df.index, cumulative_deposits, where='post', label='Einzahlungen', linestyle='--')
        ax.fill_between(self.df.index, cumulative_deposits, step='post', color='orange', alpha=0.3)

    def _plot_costs(self, ax):
        """
        Plots the costs incurred over time.

        Parameters:
            ax (Axes): The matplotlib axes to plot on
        """
        total_costs = self.df2['Total Direct Costs'].copy()

        # Sum the last two rows to account for closing costs after the end of the term
        total_costs.iloc[-1] += total_costs.iloc[-2]
        total_costs = total_costs[:-1]  # Remove the last row

        # Adjust index to match dimensions
        total_costs = total_costs.reindex(self.df.index)

        ax.step(self.df.index, total_costs, where='post', label='Kosten', linestyle='--', color='red')
        ax.fill_between(self.df.index, total_costs, step='post', color='red', alpha=0.3)

    def _plot_benchmark(self, ax):
        """
        Plots the benchmark investment performance over time.

        Parameters:
            ax (Axes): The matplotlib axes to plot on
        """
        ax.step(self.df.index, self.df['Benchmark Balance'], where='post', label='Benchmark (→ Max. Performance o. Kosten)', linestyle='--', color='green')
        ax.fill_between(self.df.index, self.df['Benchmark Balance'], self.df['Month End'], step='post', color='green', alpha=0.3)

    def _set_axes_and_title(self, ax):
        """
        Sets the axes labels and title for the plot.

        Parameters:
            ax (Axes): The matplotlib axes to set labels and title for
        """
        ax.set_title('Entwicklung der Geldanlage über die Zeit')
        ax.set_ylabel('Betrag (in €)')
        ax.set_xlim(0, len(self.df) - 0.5)

        # Different scaling of x-axis depending on duration
        if self.parameters.duration <= 4:
            ax.set_xlabel('Laufzeit (in Monaten)')
            major_locator = MultipleLocator(3)  # Main interval: every quarter
            minor_locator = MultipleLocator(1)  # Sub-interval: every month
            ax.xaxis.set_major_locator(major_locator)
            ax.xaxis.set_minor_locator(minor_locator)
        elif self.parameters.duration > 4 and self.parameters.duration < 8:
            ax.set_xlabel('Laufzeit (in Monaten)')
            major_locator = MultipleLocator(12)  # Main interval: every 12 months (1 year)
            minor_locator = MultipleLocator(3)  # Sub-interval: every 3 months
            ax.xaxis.set_major_locator(major_locator)
            ax.xaxis.set_minor_locator(minor_locator)
        else:
            ax.set_xlabel('Laufzeit (in Jahren)')
            ax.set_xticks(np.arange(0, len(self.df), 12))
            ax.set_xticklabels(np.arange(0, len(self.df) // 12 + 1))

    def _add_legend(self, ax):
        """
        Adds a legend to the plot.

        Parameters:
            ax (Axes): The matplotlib axes to add the legend to
        """
        investment_text = self._create_investment_text()
        results_text = self._create_results_text()
        ax.text(0.02, 0.98, investment_text + "\n\n" + results_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    def _create_investment_text(self):
        """
        Creates the text for the investment parameters to be displayed in the plot legend.

        Returns:
            str: The formatted investment parameters text
        """
        initial_savings_rate = self.df.iloc[1]['Monthly Savings']
        text = (
            r"$\bf{Kennzahlen\ der\ Anlage}$"
            f"\nStartdatum: {self.parameters.start_date.strftime('%d.%m.%Y')}"
            f"\nErsteinzahlung: {self.parameters.initial_investment:.2f}€"
            f"\nLaufzeit: {self.parameters.duration} Jahre"
            f"\nAnf. Sparrate: {initial_savings_rate:.2f}€"
            f"\nDynamik: {self.parameters.dynamic_increase:.2f}%"
            f"\nAngenommene Rendite: {self.parameters.return_rate}% p.a."
            f"\nErtragsausschüttung: {['jährlich', 'quartalsweise', 'monatlich'][self.parameters.yield_interval - 1]}"
            f"\nEinzahlungsgebühr (prozentual): {self.parameters.deposit_fee_percentage}%"
            f"\nEinzahlungsgebühr (fix): {self.parameters.deposit_fee_fixed}€"
        )

        if self.parameters.annual_management_fee_percentage > 0:
            fee_payment_text = 'anteilig am Monatsende' if self.parameters.fee_payment_option == 1 else 'am Jahresende'
            text += f"\nJährliche Verwaltungskosten: {self.parameters.annual_management_fee_percentage}% des Anlagevermögens"
            text += f"\nEntnahme der Verwaltungskosten: {fee_payment_text}"

        if self.parameters.seed is not None:
            text += f"\nVerwendeter Seed-Wert: {self.parameters.seed}"

        text += f"\nAbschlusskosten (prozentual): {self.parameters.closing_costs_percentage}%"
        text += f"\nAbschlusskosten (fix): {self.parameters.closing_costs_fixed}€"

        return text

    def _create_results_text(self):
        """
        Creates the text for the results to be displayed in the plot legend.

        Returns:
            str: The formatted results text
        """
        last_savings_rate = self.df.iloc[-2]['Monthly Savings']
        deposits = self.calculator.total_deposits
        benchmark_return = self.df.iloc[-1]['Benchmark Yield Distribution']
        final_benchmark_assets = self.df.iloc[-1]['Benchmark Balance']
        roi_benchmark = ((final_benchmark_assets - deposits) / deposits) * 100
        total_return = self.calculator.total_return
        final_assets = self.df.iloc[-1]['Month End']
        deposit_fees = self.calculator.total_deposit_fees
        management_fees = self.df.iloc[-1]['Costs']
        closing_costs = self.df.iloc[-1]['Fees']
        direct_costs = deposit_fees + management_fees + closing_costs
        indirect_costs = final_benchmark_assets - final_assets - direct_costs
        total_costs = direct_costs + indirect_costs
        loss = (total_costs / benchmark_return) * 100
        roi_before_tax = ((final_assets - deposits) / deposits) * 100
        tax = round((final_assets - deposits) / 100 * 26.375, 2)
        roi_after_tax = ((final_assets - tax - deposits) / deposits) * 100

        text = r"$\bf{Ergebnisse}$"
        text += f"\nEnddatum: {self.calculator.end_date.strftime('%d.%m.%Y')}"
        if self.parameters.dynamic_increase > 0:
            text += f"\nLetzte Sparrate: {last_savings_rate:.2f}€"
        if self.parameters.market_volatility > 0:
            text += f"\nDurchschnittliche Rendite: {round(self.df['Return Rate'].mean(), 2)}% (p.a.)"
        text += (
            f"\nSumme der Einzahlungen: {deposits:.2f}€"
            f"\nMaximale Summe der Kapitalerträge (Benchmark): {benchmark_return:.2f}€"
            f"\nMaximales Endvermögen (Benchmark): {final_benchmark_assets:.2f}€"
            # f"\nBenchmark-ROI: {roi_benchmark:.2f}%"
            f"\nTatsächliche Summe der Kapitalerträge: {total_return:.2f}€"
            f"\nTatsächliches Anlagevermögen zum Ende der Laufzeit: {final_assets + closing_costs:.2f}€"
            # f"\nSumme der Einzahlungsgebühren (prozentual+fix): {deposit_fees:.2f}€"
            # f"\nSumme der jährlichen Verwaltungskosten (prozentual): {management_fees:.2f}€"
            # f"\nSumme der Abschlusskosten (prozentual+fix): {closing_costs:.2f}€"
            f"\nSumme der direkten Kosten: {direct_costs:.2f}€"
            f"\nSumme der indirekten Kosten: {indirect_costs:.2f}€"
            f"\nSumme aller Kosten (direkt+indirekt): {total_costs: .2f}€"
        )
        if total_costs > 0:
            text += f"\nAnteil der indirekten Kosten an Gesamtkosten: {indirect_costs/total_costs*100:.2f}%"
        text += (
            f"\nProzentualer Gesamtverlust durch Kosten: {loss: .2f}%"
            f"\nEndkontostand nach Abschlusskosten: {final_assets:.2f}€"
            f"\nKapitalertragssteuer inkl. Solidaritätszuschlag (26.375%): {tax:.2f}€"
            # f"\nROI vor Steuern: {roi_before_tax:.2f}%"
            f"\nEndvermögen nach Steuern: {final_assets - tax:.2f}€"
            # f"\nROI nach Steuern: {roi_after_tax:.2f}%"
        )
        return text


# ------------------------------------------------- Program Execution -------------------------------------------------

class InvestmentOptimizer:
    def __init__(self):
        """
        Initializes the Investment Optimizer with default parameters and settings.
        """
        self.parameters = None
        self.calculator = None
        self.visualizer = None
        self.results = []

        self.param_names = {
            'initial_investment': 'Ersteinzahlung (€)',
            'start_date': 'Startdatum',
            'duration': 'Laufzeit (Jahre)',
            'monthly_savings': 'Anf. Sparrate (€)',
            'initial_monthly_savings': 'Anf. Sparrate (€)',
            'dynamic_increase': 'Dynamik (%)',
            'return_rate': 'Rendite (%)',
            'market_volatility': 'Marktvolatilität (%)',
            'yield_interval': 'Renditeausschüttung (monatl./quartalsweise/jährl.)',
            'deposit_fee_percentage': 'Einzahlungsgebühren (prozentual) (%)',
            'deposit_fee_fixed': 'Einzahlungsgebühren (fix) (€)',
            'annual_management_fee_percentage': 'Jährl. Verwaltungskosten (%) (€)',
            'fee_payment_option': 'Verrechnungsoption (monatl./jährl.)',
            'closing_costs_percentage': 'Abschlusskosten (prozentual) (%)',
            'closing_costs_fixed': 'Abschlusskosten (fix) (€)',
            'seed': 'Seed-Wert'
        }

        self.result_names = {
            'start_date': 'Startdatum',
            'end_date': 'Enddatum',
            'last_savings_rate': 'Höhe der letzten Sparrate (€)',
            'total_deposits': 'Summe der Einzahlungen über die Laufzeit (€)',
            'benchmark_return': 'Maximale Summe der Kapitalerträge über die Laufzeit (Benchmark) (€)',
            'final_benchmark_assets': 'Maximales Endvermögen (Benchmark) (€)',
            'roi_benchmark': 'Benchmark-ROI (Return on Investment) (%)',
            'total_return': 'Summe der Kapitalerträge über die Laufzeit (€)',
            'total_deposit_fees': 'Summe der Einzahlungsgebühren (prozentual+fix) (€)',
            'total_management_fees': 'Summe der jährlichen Verwaltungskosten (prozentual) (€)',
            'closing_costs': 'Abschlusskosten (prozentual+fix) (€)',
            'total_direct_costs': 'Summe aller (direkten) Kosten (€)',
            'indirect_costs': 'Indirekte Kosten (€)',
            'total_costs': 'Gesamtkosten (€)',
            'loss': 'Gesamtverlust durch Kosten (%)',
            'indirect_share': 'Anteil der indirekten Kosten an Gesamtkosten (%)',
            'final_value_after_closing_costs': 'Wert der Geldanlage zum Ende der Laufzeit (nach Abschlusskosten) (€)',
            'roi_before_tax': 'ROI vor Steuern (%)',
            'tax': 'Kapitalertragssteuer inkl. Solidaritätszuschlag (26.375%) (€)',
            'final_value_after_tax': 'Endvermögen nach Steuern (€)',
            'roi_after_tax': 'ROI nach Steuern (%)'
        }

    def run(self):
        """
        Runs the investment optimizer program, prompting the user for input and displaying results.
        """
        self._print_welcome_message()
        while True:
            self.parameters = self._get_user_input()
            self.calculator = InvestmentCalculator(self.parameters)
            df = self.calculator.calculate()
            self._print_results(df)
            self.visualizer = InvestmentVisualizer(df, self.parameters, self.calculator)
            fig = self.visualizer.visualize()

            result = {
                "parameters": self.parameters,
                "df": df,
                "figure": fig,
                "start_date": self.parameters.start_date,
                "end_date": self.calculator.end_date,
                "last_savings_rate": df.iloc[-2]['Monthly Savings'],
                "total_deposits": self.calculator.total_deposits,
                "benchmark_return": df.iloc[-1]['Benchmark Yield Distribution'],
                "final_benchmark_assets": df.iloc[-1]['Benchmark Balance'],
                "roi_benchmark": ((df.iloc[-1]['Benchmark Balance'] - self.calculator.total_deposits) / self.calculator.total_deposits) * 100,
                "total_return": self.calculator.total_return,
                "total_deposit_fees": self.calculator.total_deposit_fees,
                "total_management_fees": df.iloc[-1]['Costs'],
                "final_value_before_closing_costs": df.iloc[-1]['Month End'] + df.iloc[-1]['Fees'],
                "closing_costs": df.iloc[-1]['Fees'],
                "total_direct_costs": self.calculator.total_deposit_fees + df.iloc[-1]['Fees'],
                "indirect_costs": self.calculator.total_deposit_fees + df.iloc[-1]['Costs'] + df.iloc[-1]['Fees'],
                "total_costs": (self.calculator.total_deposit_fees + df.iloc[-1]['Fees']) + (self.calculator.total_deposit_fees + df.iloc[-1]['Costs'] + df.iloc[-1]['Fees']),
                "loss": (((self.calculator.total_deposit_fees + df.iloc[-1]['Fees']) + (self.calculator.total_deposit_fees + df.iloc[-1]['Costs'] + df.iloc[-1]['Fees'])) / df.iloc[-1]['Benchmark Yield Distribution']) * 100,
                "indirect_share": ((self.calculator.total_deposit_fees + df.iloc[-1]['Costs'] + df.iloc[-1]['Fees']) / ((self.calculator.total_deposit_fees + df.iloc[-1]['Fees']) + (self.calculator.total_deposit_fees + df.iloc[-1]['Costs'] + df.iloc[-1]['Fees']))) * 100,
                "roi_before_tax": ((df.iloc[-1]['Month End'] - self.calculator.total_deposits) / self.calculator.total_deposits) * 100,
                "tax": (df.iloc[-1]['Month End'] - self.calculator.total_deposits) * 0.26375,
            }
            result["final_value_after_tax"] = df.iloc[-1]['Month End'] + df.iloc[-1]['Fees'] - result["tax"]
            result["roi_after_tax"] = ((result["final_value_after_tax"] - self.calculator.total_deposits) / self.calculator.total_deposits) * 100

            self.results.append(result)

            save_parameters(self.parameters, USER_PARAMS_FILE)

            if len(self.results) > 1:
                self._compare_results(self.results[-2], self.results[-1])

            another_run = input("\nMöchten Sie einen weiteren Durchlauf zum Vergleich starten? (j/n): ")
            if another_run.lower() != 'j':
                break

        print("\033[94m\nProgramm beendet. Vielen Dank für die Nutzung des Investment Optimizers!\033[0m")

    def _print_welcome_message(self):
        """
        Prints a welcome message and an overview of the program's functionality.
        """
        print("\033[94m========================\033[0m")
        print("\033[94m| Investment Optimizer |\033[0m")
        print("\033[94m========================\033[0m")

        print("\nDieses Programm berechnet die Entwicklung einer Geldanlage über einen bestimmten Zeitraum unter Berücksichtigung verschiedener Parameter.")
        print("    \033[32mAnlageparameter\033[0m: Anlagesumme, Laufzeit, Sparrate, Dynamik (jährliche Erhöhung der Sparrate), Rendite (p.a.), Intervall der Renditeausschüttung")
        print("    \033[31mKostenparameter\033[0m: prozentuale Einzahlungsgebühr (% der Einzahlung), fixe Einzahlungsgebühr (von Einzahlungen abgezogener Festbetrag),")
        print("                     jährliche Verwaltungskosten (wahlweise zum Jahresende oder anteilig zum Monatsende vom Anlagevermögen entnommener Prozentsatz),")
        print("                     prozentuale Abschlusskosten (% des Endvermögens), fixe Abschlusskosten (vom Endvermögen abgezogener Festbetrag)")
        print("\nNeben Geldanlagen mit festgelegter Rendite (Zinsen) können auch volatile Erträge (wie etwa bei Aktienfonds oder ETFs) simuliert werden.")
        print("Die Ergebnisse werden sowohl tabellarisch als auch grafisch über die gesamte Anlagelaufzeit ausgegeben.")

    def _get_user_input(self):
        """
        Prompts the user for investment parameters and returns an InvestmentParameters object.

        Returns:
            InvestmentParameters: The user-defined investment parameters
        """
        print(
            "\nBitte geben Sie die folgenden Parameter ein (oder drücken Sie 'Enter' zur Übernahme der vorgeschlagenen Werte):")

        # Investment parameters
        print("\n\033[32mAnlageparameter\033[0m:")
        print("----------------")
        params = load_parameters(USER_PARAMS_FILE) if self.parameters else load_parameters(DEFAULT_PARAMS_FILE)
        if params is None:
            params = {
                'initial_investment': 10000,
                'duration': 5,
                'start_date': datetime.now().date() + timedelta(days=1),
                'initial_monthly_savings': 100,
                'dynamic_increase': 5,
                'return_rate': 7,
                'market_volatility': 0,
                'yield_interval': 2,
                'deposit_fee_percentage': 0.5,
                'deposit_fee_fixed': 0,
                'annual_management_fee_percentage': 0.5,
                'fee_payment_option': 2,
                'closing_costs_percentage': 0.5,
                'closing_costs_fixed': 0,
                'seed': None
            }

        initial_investment = self._get_float_input(f"Ersteinzahlung (in €) [Vorschlag: {params['initial_investment']}]: ",
            min_value=0, default=params['initial_investment'])
        duration = self._get_int_input(f"Laufzeit (in Jahren) [Vorschlag: {params['duration']}]: ",
            min_value=1, max_value=100, default=params['duration'])
        start_date = self._get_date_input(f"Startdatum (TT.MM.JJJJ) [Vorschlag: {params['start_date'].strftime('%d.%m.%Y')}]: ",
            default=params['start_date'])
        monthly_savings = self._get_float_input(f"Monatliche Sparrate (in €) [Vorschlag: {params['initial_monthly_savings']}]: ",
            min_value=0, default=params['initial_monthly_savings'])
        dynamic_increase = self._get_float_input(f"Jährliche Erhöhung der Sparrate (in %) [Vorschlag: {params['dynamic_increase']}]: ",
            min_value=0, max_value=100, default=params['dynamic_increase'])
        investment_type = self._get_int_input(f"Feste Rendite (Zinsen) oder variabler Ertrag (Kursanstieg)? '1'=fest, '2'=variabel [Vorschlag: {'2' if params['market_volatility'] > 0 else '1'}]: ",
            min_value=1, max_value=2, default=2 if params['market_volatility'] > 0 else 1
        )

        seed = None
        if investment_type == 1:
            return_rate = self._get_float_input(f"Rendite (in % p.a.) [Vorschlag: {params['return_rate']}]: ",
                                                  min_value=-100, max_value=100, default=params['return_rate'])
            market_volatility = 0
            yield_interval = self._get_int_input(f"Ertragsintervall (Zinsauszahlung) (1: jährlich, 2: quartalsweise, 3: monatlich) [Vorschlag: {params['yield_interval']}]: ",
                                                        min_value=1, max_value=3, default=params['yield_interval'])
        else:
            return_rate = self._get_float_input(f"Angenommene durchschnittliche Rendite (in % p.a.) [Vorschlag: {params['return_rate']}]: ",
                min_value=-100, max_value=100, default=params['return_rate'])
            yield_interval = 3
            market_volatility = self._get_float_input(f"Marktvolatilität (0-100) [Vorschlag: {params['market_volatility'] or 50}]: ",
                min_value=0, max_value=100, default=params['market_volatility'] or 50)
            use_seed = self._get_int_input("Möchten Sie einen Seed-Wert für die Zufallszahlengenerierung verwenden? (1: Ja, 0: Nein) [Vorschlag: 1]: ",
                min_value=0, max_value=1, default=1)
            if use_seed:
                seed = self._get_int_input(f"Geben Sie den Seed-Wert ein (Ganzzahl) [Vorschlag: {params['seed'] or 42}]: ",
                    default=params['seed'] or 42)

        # Cost parameters
        print("\n\033[31mKostenparameter\033[0m:")
        print("----------------")
        deposit_fee_percentage = self._get_float_input(f"Prozentuale Einzahlungsgebühr (in % p.a.) [Vorschlag: {params['deposit_fee_percentage']}]: ",
            min_value=0, max_value=100, default=params['deposit_fee_percentage'])
        deposit_fee_fixed = self._get_float_input(f"Fixe Einzahlungsgebühr (in € p.a.) [Vorschlag: {params['deposit_fee_fixed']}]: ",
            min_value=0, default=params['deposit_fee_fixed'])
        annual_management_fee_percentage = self._get_float_input(f"Jährliche Verwaltungskosten (in % vom Anlagevermögen) [Vorschlag: {params['annual_management_fee_percentage']}]: ",
            min_value=0, default=params['annual_management_fee_percentage'])
        if annual_management_fee_percentage > 0:
            fee_payment_option = self._get_int_input(f"Verrechnungsoption für die Verwaltungskosten (1: monatlich, 2: jährlich) [Vorschlag: {params['fee_payment_option']}]: ",
                min_value=1, max_value=2, default=params['fee_payment_option'])
        else:
            fee_payment_option = params['fee_payment_option']
        closing_costs_percentage = self._get_float_input(f"Prozentuale Abschlusskosten (in %) [Vorschlag: {params['closing_costs_percentage']}]: ",
            min_value=0, max_value=100, default=params['closing_costs_percentage'])
        closing_costs_fixed = self._get_float_input(f"Fixe Abschlusskosten (in €) [Vorschlag: {params['closing_costs_fixed']}]: ",
            min_value=0, default=params['closing_costs_fixed'])

        print(f"\nFür die Berechnung der Einzahlungsgebühr wird von {deposit_fee_percentage}% des Anlagebetrags sowie zusätzlich einem Fixbetrag von {deposit_fee_fixed}€ ausgegangen.")
        if annual_management_fee_percentage > 0 and fee_payment_option == 1:
            print(f"Die Verwaltungskosten i.H.v. {annual_management_fee_percentage}% (p.a.) des aktuellen Anlagevermögens wird anteilig zum Ende jedes Monats verbucht.")
        elif annual_management_fee_percentage > 0 and fee_payment_option == 2:
            print(f"Die Verwaltungskosten i.H.v. {annual_management_fee_percentage}% (p.a.) des aktuellen Anlagevermögens werden jeweils am Jahresende verbucht.")
        print(f"Zudem werden prozentuale Abschlusskosten i.H.v. {closing_costs_percentage}% vom Endvermögen und ggf. ein Fixbetrag von {closing_costs_fixed}€ abgezogen.")
        print(f"Außerdem wird die Kapitalertragssteuer i.H.v. 26.375% (inkl. Solidaritätszuschlag) berücksichtigt.")

        return InvestmentParameters(initial_investment, duration, monthly_savings, dynamic_increase, return_rate,
                                    market_volatility, yield_interval, deposit_fee_percentage, deposit_fee_fixed,
                                    annual_management_fee_percentage, fee_payment_option, closing_costs_percentage,
                                    closing_costs_fixed, seed, start_date)

    def _get_date_input(self, prompt, default=None):
        """
        Prompts the user for a date input and validates the format.

        Parameters:
            prompt (str): The prompt message to display to the user
            default (date, optional): The default date to return if the user presses Enter

        Returns:
            date: The validated date input from the user
        """
        while True:
            user_input = input(prompt)
            if user_input == "" and default is not None:
                return default
            try:
                return datetime.strptime(user_input, "%d.%m.%Y").date()
            except ValueError:
                print("\033[91mFehler: Bitte geben Sie ein gültiges Datum im Format TT.MM.JJJJ ein.\033[0m")

    def _get_float_input(self, prompt, min_value=None, max_value=None, default=None):
        """
        Prompts the user for a float input and validates the value.

        Parameters:
            prompt (str): The prompt message to display to the user
            min_value (float, optional): The minimum acceptable value
            max_value (float, optional): The maximum acceptable value
            default (float, optional): The default value to return if the user presses Enter

        Returns:
            float: The validated float input from the user
        """
        while True:
            user_input = input(prompt)
            if user_input == "" and default is not None:
                return default
            try:
                value = float(user_input)
                if min_value is not None and value < min_value:
                    print(f"\033[91mFehler: Der Wert muss mindestens {min_value} sein.\033[0m")
                elif max_value is not None and value > max_value:
                    print(f"\033[91mFehler: Der Wert darf höchstens {max_value} sein.\033[0m")
                else:
                    return value
            except ValueError:
                print("\033[91mFehler: Bitte geben Sie eine gültige Zahl ein.\033[0m")

    def _get_int_input(self, prompt, min_value=None, max_value=None, default=None):
        """
        Prompts the user for an integer input and validates the value.

        Parameters:
            prompt (str): The prompt message to display to the user
            min_value (int, optional): The minimum acceptable value
            max_value (int, optional): The maximum acceptable value
            default (int, optional): The default value to return if the user presses Enter

        Returns:
            int: The validated integer input from the user.
        """
        while True:
            user_input = input(prompt)
            if user_input == "" and default is not None:
                return default
            try:
                value = int(user_input)
                if min_value is not None and value < min_value:
                    print(f"\033[91mFehler: Der Wert muss mindestens {min_value} sein.\033[0m")
                elif max_value is not None and value > max_value:
                    print(f"\033[91mFehler: Der Wert darf höchstens {max_value} sein.\033[0m")
                else:
                    return value
            except ValueError:
                print("\033[91mFehler: Bitte geben Sie eine gültige Ganzzahl ein.\033[0m")

    def _print_results(self, df):
        """
        Prints the results of the investment calculations in a formatted manner.

        Parameters:
            df (DataFrame): The DataFrame containing the investment results
        """
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        pd.options.display.float_format = '{:.2f}'.format

        print("\nEntwicklung der Geldanlage über die Laufzeit:")
        print("-----------------------------------------------")
        print(df)

        print("\nErgebnisse der Berechnung:")
        print("--------------------------")
        print(f"Enddatum des Investments: \033[94m{self.calculator.end_date.strftime('%d.%m.%Y')}\033[0m")

        # Deposits across time
        dynamic_increase = self.parameters.dynamic_increase
        if dynamic_increase > 0:
            last_savings_rate = df.iloc[-2]['Monthly Savings']
            print(f"Höhe der letzten Sparrate: \033[32m{last_savings_rate:.2f}€\033[0m")
        deposits = self.calculator.total_deposits
        print(f"Summe der Einzahlungen über die Laufzeit: \033[32m{deposits:.2f}€\033[0m")

        # Benchmark Scenario Analysis
        benchmark_return = df.iloc[-1]['Benchmark Yield Distribution']
        print(f"\nMaximale Summe der Kapitalerträge über die Laufzeit (Benchmark): \033[32m{benchmark_return:.2f}€\033[0m")
        final_benchmark_assets = df.iloc[-1]['Benchmark Balance']
        print(f"Maximales Endvermögen (Benchmark): \033[32m{final_benchmark_assets:.2f}€\033[0m")
        roi_benchmark = ((final_benchmark_assets - deposits) / deposits) * 100
        print(f"Benchmark-ROI (Return on Investment): \033[32m{roi_benchmark:.2f}%\033[0m")

        # Returns vs. Costs Analysis
        total_return = self.calculator.total_return
        print(f"\nTatsächliche Summe der Kapitalerträge über die Laufzeit: \033[32m{total_return:.2f}€\033[0m")
        final_assets = df.iloc[-1]['Month End']
        print(f"Tatsächlicher Wert der Geldanlage zum Ende der Laufzeit: \033[32m{final_assets:.2f}€\033[0m")
        print(f"ROI der Geldanlage: \033[32m{((final_assets - deposits) / deposits) * 100:.2f}%\033[0m")

        deposit_fees = self.calculator.total_deposit_fees
        print(f"\nSumme der Einzahlungsgebühren (prozentual+fix): \033[91m{deposit_fees:.2f}€\033[0m")
        management_fees = df.iloc[-1]['Costs']
        print(f"Summe der jährlichen Verwaltungskosten (prozentual): \033[91m{management_fees:.2f}€\033[0m")
        closing_costs = df.iloc[-1]['Fees']
        print(f"Abschlusskosten (prozentual+fix): \033[91m{closing_costs:.2f}€\033[0m")
        direct_costs = deposit_fees + management_fees + closing_costs
        print(f"Summe der direkten Kosten: \033[91m{direct_costs:.2f}€\033[0m")
        indirect_costs = final_benchmark_assets - final_assets - direct_costs
        print(f"Summe der indirekten Kosten: \033[91m{indirect_costs:.2f}€ (=entgangene Kapitalerträge)\033[0m")
        total_costs = direct_costs + indirect_costs
        print(f"Summe aller Kosten (direkt+indirekt): \033[91m{total_costs:.2f}€\033[0m")
        if total_costs > 0:
            print(f"Anteil der indirekten Kosten an Gesamtkosten: \033[91m{indirect_costs/total_costs*100:.2f}%\033[0m")
        loss = (total_costs / benchmark_return) * 100
        print(f"Prozentualer Gesamtverlust durch Kosten: \033[91m{loss:.2f}%\033[0m")

        # Additional metrics
        if self.parameters.market_volatility > 0:
            print(f"\nDurchschnittliche Rendite über die Laufzeit: \033[32m{df['Return Rate'].iloc[1:-1].mean():.2f}% (p.a.)\033[0m")
            if self.parameters.seed is not None:
                print(f"Verwendeter Seed-Wert: \033[32m{self.parameters.seed}\033[0m")

        # Tax deduction
        tax = (final_assets - deposits) / 100 * 26.375
        print(f"\nKapitalertragssteuer inkl. Solidaritätszuschlag (26.375%): \033[31m{tax:.2f}€\033[0m")
        print(f"Endvermögen nach Steuern: \033[94m{final_assets - tax:.2f}€\033[0m")
        print(f"ROI nach Steuern: \033[94m{((final_assets - tax - deposits) / deposits) * 100:.2f}%\033[0m")


    # --------------------------------------------- Investment Comparison ---------------------------------------------

    def _compare_parameters(self, prev_params, curr_params):
        """
        Compares the investment parameters of two models and prints the differences.

        Parameters:
            prev_params (InvestmentParameters): The previous investment parameters
            curr_params (InvestmentParameters): The current investment parameters
        """
        print("\nGegenüberstellung der Parameter:")
        print("--------------------------------")
        for key, value in vars(curr_params).items():
            if key == 'initial_monthly_savings':
                continue
            prev_value = getattr(prev_params, key)
            if prev_value != value:
                print(f"{key}:")
                print(f"  Vorheriger Wert: {prev_value}")
                print(f"  Aktueller Wert:  {value}")
                if isinstance(value, (int, float)) and isinstance(prev_value, (int, float)):
                    diff = value - prev_value
                    print(f"  Änderung:        {diff:+.2f} ({'↑' if diff > 0 else '↓'})")
            print()

    def _compare_results(self, prev_result, curr_result):
        """
        Compares the results of two investment models and prints the differences.

        Parameters:
            prev_result (dict): The results of the previous investment model
            curr_result (dict): The results of the current investment model
        """
        # Parameter comparison
        param_table = []
        for key, value in vars(curr_result['parameters']).items():
            prev_value = getattr(prev_result['parameters'], key)
            param_name = self.param_names.get(key, key)
            if key == 'start_date':
                prev_value = prev_value.strftime('%d.%m.%Y') if isinstance(prev_value, (date, datetime)) else prev_value
                value = value.strftime('%d.%m.%Y') if isinstance(value, (date, datetime)) else value
                diff = "Geändert" if prev_value != value else ""
            elif prev_value != value:
                if isinstance(value, (int, float)) and isinstance(prev_value, (int, float)):
                    diff = f"{value - prev_value:+.2f}"
                    prev_value = f"{prev_value:.2f}"
                    value = f"{value:.2f}"
                else:
                    diff = "Geändert"
                    prev_value = str(prev_value)
                    value = str(value)
            else:
                diff = ""
                if isinstance(value, (int, float)):
                    prev_value = f"{prev_value:.2f}"
                    value = f"{value:.2f}"
                else:
                    prev_value = str(prev_value)
                    value = str(value)
            param_table.append([param_name, prev_value, value, diff])

        print("\nGegenüberstellung der Parameter:")
        print(tabulate(param_table, headers=["Parameter", "Modell 1", "Modell 2", "Differenz"], tablefmt="grid"))

        # Result comparison
        result_table = []
        result_keys = [
            'start_date', 'end_date', 'last_savings_rate', 'total_deposits', 'benchmark_return', 'final_benchmark_assets',
            'roi_benchmark', 'total_return', 'total_deposit_fees', 'total_management_fees', 'closing_costs',
            'total_direct_costs', 'indirect_costs', 'total_costs', 'loss', 'indirect_share',
            'final_value_after_closing_costs', 'roi_before_tax', 'tax', 'final_value_after_tax', 'roi_after_tax'
        ]

        for key in result_keys:
            prev_value = prev_result.get(key, 0)
            curr_value = curr_result.get(key, 0)
            if key in ['start_date', 'end_date']:
                prev_value = prev_value.strftime('%d.%m.%Y') if isinstance(prev_value, (date, datetime)) else prev_value
                curr_value = curr_value.strftime('%d.%m.%Y') if isinstance(curr_value, (date, datetime)) else curr_value
                diff = "Geändert" if prev_value != curr_value else ""
            elif isinstance(prev_value, (int, float)) and isinstance(curr_value, (int, float)):
                diff = curr_value - prev_value
                result_table.append([self.result_names[key], f"{prev_value:.2f}", f"{curr_value:.2f}", f"{diff:+.2f}"])
                continue
            else:
                diff = ""
            result_table.append([self.result_names[key], str(prev_value), str(curr_value), diff])

        print("\nVergleich der Ergebnisse:")
        print(tabulate(result_table, headers=["Kennzahl", "Modell 1", "Modell 2", "Differenz"], tablefmt="grid"))

        # Visualizing the comparison
        plt.figure(figsize=(12, 6))

        # Plot and fill for Model 1
        plt.plot(prev_result['df'].index, prev_result['df']['Month End'], label='Modell 1', color='blue', linewidth=2)
        plt.fill_between(prev_result['df'].index, prev_result['df']['Month End'], color='blue', alpha=0.1)

        # Plot and fill for Model 2
        plt.plot(curr_result['df'].index, curr_result['df']['Month End'], label='Modell 2', color='red', linewidth=2,
                 alpha=0.7)
        plt.fill_between(curr_result['df'].index, curr_result['df']['Month End'], color='red', alpha=0.1)

        plt.title('Vergleich der Vermögensentwicklung')
        plt.xlabel('Monate')
        plt.ylabel('Anlagewert (€)')
        plt.legend()
        plt.grid(True)

        # x-axis refers to longest duration
        max_duration = max(len(prev_result['df']), len(curr_result['df']))
        plt.xlim(0, max_duration)

        plt.tight_layout()
        plt.show()

# --------------------------------------------------- Program Call ---------------------------------------------------

if __name__ == "__main__":
    optimizer = InvestmentOptimizer()
    optimizer.run()