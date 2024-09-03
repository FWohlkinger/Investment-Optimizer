##################################
###    Investment Optimizer    ###
##################################
### Author: Florian Wohlkinger ###
### Date: 03.09.2024           ###
### Version: 2.0.0             ###
##################################

# Changes in this version:
#   - Implementation of object-oriented programming
#   - Error messages for implausible inputs
#   - Seed value for the random number generator
#   - Specified ROI before/after taxes
#   - Specified default values
#   - Implemented program loop + scenario comparison

# ==================================================================================================================== #

# Import modules
import pandas as pd                             # for Data Frame
import numpy as np                              # for normally distributed random variable
import matplotlib.pyplot as plt                 # for visualization
from matplotlib.ticker import MultipleLocator   # for visualization (axis labeling)
import json                                     # for saving the investment parameters
from tabulate import tabulate                   # for scenario comparison

# ----------------------------------------------- Save & Load Parameters -----------------------------------------------

def load_parameters(file_name):
    try:
        with open(file_name, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def save_parameters(parameters, file_name):
    params_dict = vars(parameters)
    params_dict['initial_monthly_savings'] = parameters.initial_monthly_savings
    with open(file_name, 'w') as f:
        json.dump(params_dict, f, indent=2)

# Define filenames
DEFAULT_PARAMS_FILE = 'default_params.json'
USER_PARAMS_FILE = 'user_params.json'

# ------------------------------------------------ InvestmentParameters ------------------------------------------------
class InvestmentParameters:
    def __init__(self, initial_investment, duration, monthly_savings, dynamic_increase, interest_rate,
                 interest_volatility, payout_interval,
                 management_fee_variable, management_fee_fixed, closing_costs_variable, closing_costs_fixed, seed=None):
        self.initial_investment = initial_investment
        self.duration = duration
        self.initial_monthly_savings = monthly_savings
        self.monthly_savings = monthly_savings
        self.dynamic_increase = dynamic_increase
        self.interest_rate = interest_rate
        self.interest_volatility = interest_volatility
        self.payout_interval = payout_interval
        self.management_fee_variable = management_fee_variable  # Percentage deduction from deposits
        self.management_fee_fixed = management_fee_fixed        # Fixed amount --> deduction from monthly deposits
        self.closing_costs_variable = closing_costs_variable    # Percentage deduction from final assets (before taxes)
        self.closing_costs_fixed = closing_costs_fixed          # Fixed amount --> deduction from final assets (before taxes)
        self.seed = int(seed) if seed is not None else None


# --------------------------------------------------- InvestmentCalculator ---------------------------------------------------
class InvestmentCalculator:
    def __init__(self, parameters):
        self.parameters = parameters
        self.data = []
        self.total_deposits = parameters.initial_investment
        self.total_interest = 0
        self.total_management_fees = 0
        # Set seed value
        if self.parameters.seed is not None:
            np.random.seed(self.parameters.seed)

    def calculate(self):
        self.data = []
        balance = self.parameters.initial_investment
        monthly_savings = self.parameters.initial_monthly_savings

        # Row 0 before start of duration ("Month 0") --> Duration starts with initial investment
        self.data.append([0, 0, 0, 0, 0, 0, 0, 0, 0, self.parameters.initial_investment, 0])

        # Loop for calendar year
        for year in range(1, self.parameters.duration + 1):
            annual_interest = 0
            previous_month_interest_rate = self.parameters.interest_rate

            # Loop for individual months
            for month in range(1, 13):
                monthly_interest_rate = self._calculate_interest_rate(previous_month_interest_rate)
                previous_month_interest_rate = monthly_interest_rate

                management_fees, investment_rate = self._calculate_costs(monthly_savings)

                month_start = balance
                balance += investment_rate
                monthly_interest = balance * ((monthly_interest_rate / 100) / 12)

                self._update_total_values(monthly_interest, management_fees, monthly_savings)

                self.data.append(self._create_monthly_data(year, month, monthly_interest_rate, month_start,
                                                           monthly_savings, management_fees, investment_rate,
                                                           monthly_interest, balance))

                balance, annual_interest = self._process_interest_payout(balance, annual_interest, monthly_interest,
                                                                         month)

            monthly_savings = self._update_savings_rate(year, monthly_savings)

        self._add_closing_data(balance)

        return pd.DataFrame(self.data, columns=["Year", "Month", "Interest Rate", "Month Start", "Monthly Savings",
                                                "Costs", "Total Costs", "Monthly Investment Rate", "Monthly Interest",
                                                "Month End", "Interest Payout"])

    def _calculate_interest_rate(self, previous_month_interest_rate):
        ### Procedure for "volatile interest rates": Generate a normally distributed random variable
        #   with dependency on the previous month's interest rate --> autoregressive component (AR)
        interest_stddev = self.parameters.interest_rate * self.parameters.interest_volatility / 100
        random_variable = np.random.normal(0, interest_stddev)
        AR_parameter = 0.5  # AR parameter for the AR(1) process

        if interest_stddev == 0:  # Fixed interest (no volatility)
            return self.parameters.interest_rate
        else:  # Variable interest rate --> AR model
            return self.parameters.interest_rate + AR_parameter * (
                    previous_month_interest_rate - self.parameters.interest_rate) + random_variable

    def _calculate_costs(self, monthly_savings):
        # Calculation of management fees and potentially reduced monthly investment rate
        management_fees_percentage = monthly_savings * self.parameters.management_fee_variable / 100
        management_fees = management_fees_percentage + self.parameters.management_fee_fixed
        investment_rate = monthly_savings - management_fees
        return management_fees, investment_rate

    def _update_total_values(self, monthly_interest, management_fees, monthly_savings):
        self.total_interest += monthly_interest
        self.total_deposits += monthly_savings
        self.total_management_fees += management_fees

    def _create_monthly_data(self, year, month, monthly_interest_rate, month_start, monthly_savings,
                             management_fees, investment_rate, monthly_interest, balance):
        return [year, month, monthly_interest_rate, round(month_start, 2), monthly_savings,
                management_fees, self.total_management_fees, investment_rate, round(monthly_interest, 2),
                round(balance, 2), 0]

    def _process_interest_payout(self, balance, annual_interest, monthly_interest, month):
        # Interest payout according to interval
        annual_interest += monthly_interest
        if self.parameters.payout_interval == 1 and month == 12:  # annually --> Interest payout at year-end
            self.data[-1][-1] = round(annual_interest, 2)
            balance += self.data[-1][-1]
            annual_interest = 0
        elif self.parameters.payout_interval == 2 and month % 3 == 0:  # quarterly --> Interest payout at end of quarter
            self.data[-1][-1] = round(annual_interest, 2)
            balance += self.data[-1][-1]
            annual_interest = 0
        elif self.parameters.payout_interval == 3:  # monthly
            self.data[-1][-1] = round(monthly_interest, 2)
            balance += self.data[-1][-1]
        return balance, annual_interest

    def _update_savings_rate(self, year, current_savings):
        # Potential annual increase of savings rate by the percentage of dynamic increase
        if self.parameters.dynamic_increase > 0 and year < self.parameters.duration:
            dynamic_amount = round(current_savings * (self.parameters.dynamic_increase / 100), 2)
            return round(current_savings + dynamic_amount, 2)
        return current_savings

    def _add_closing_data(self, balance):
        # Add last row at the end of the dataset --> contains the final balance including last interest payout
        final_assets = balance
        closing_costs_variable = round(final_assets * self.parameters.closing_costs_variable / 100, 2)
        closing_costs = closing_costs_variable + self.parameters.closing_costs_fixed
        final_assets_after_closing_costs = final_assets - closing_costs
        self.data.append([self.parameters.duration + 1, 0, 0, 0, 0, closing_costs,
                          self.total_management_fees + closing_costs, 0, 0,
                          final_assets_after_closing_costs, self.total_interest])


# --------------------------------------------------- Visualization ---------------------------------------------------

class InvestmentVisualizer:
    def __init__(self, df, parameters):
        self.df = df
        self.parameters = parameters

    def visualize(self):
        self.df = self.df.iloc[1:].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(10, 6))

        self._plot_asset_development(ax)
        self._plot_deposits(ax)
        self._plot_costs(ax)

        self._set_axes_and_title(ax)
        self._add_legend(ax)

        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        return fig

    def _plot_asset_development(self, ax):
        ax.step(self.df.index, self.df['Month End'], where='post', label='Vermögensstand')
        ax.fill_between(self.df.index, self.df['Month End'], step='post', color='blue', alpha=0.3)


    def _plot_deposits(self, ax):
        cumulative_deposits = self.df['Monthly Savings'].cumsum() + self.parameters.initial_investment
        ax.step(self.df.index, cumulative_deposits, where='post', label='Einzahlungen', linestyle='--')
        ax.fill_between(self.df.index, cumulative_deposits, step='post', color='orange', alpha=0.3)

    def _plot_costs(self, ax):
        ax.plot(self.df.index, self.df['Total Costs'], label='Kosten', linestyle='--', color='red')
        ax.fill_between(self.df.index, self.df['Total Costs'], step='post', color='red', alpha=0.3)

    def _set_axes_and_title(self, ax):
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
        investment_text = self._create_investment_text()
        results_text = self._create_results_text()
        ax.text(0.02, 0.98, investment_text + "\n\n" + results_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    def _create_investment_text(self):
        # Text field (first part) for the legend with the entered investment parameters
        initial_savings_rate = self.df.iloc[1]['Monthly Savings']
        text = (
            r"$\bf{Kennzahlen\ der\ Anlage}$"
            f"\nErsteinzahlung: {self.parameters.initial_investment:.2f}€"
            f"\nLaufzeit: {self.parameters.duration} Jahre"
            f"\nAnf. Sparrate: {initial_savings_rate:.2f}€"
            f"\nDynamik: {self.parameters.dynamic_increase:.2f}%"
            f"\nVerwaltungskosten (variabel): {self.parameters.management_fee_variable}%"
            f"\nVerwaltungskosten (fix): {self.parameters.management_fee_fixed}€"
            f"\nAbschlusskosten (variabel): {self.parameters.closing_costs_variable}%"
            f"\nAbschlusskosten (fix): {self.parameters.closing_costs_fixed}€"
            f"\nZinsauszahlung: {['jährlich', 'quartalsweise', 'monatlich'][self.parameters.payout_interval - 1]}"
            f"\nAngenommener Zinssatz (p.a.): {self.parameters.interest_rate}%"
        )
        if self.parameters.seed is not None:
            text += f"\nVerwendeter Seed-Wert: {self.parameters.seed}"
        return text

    def _create_results_text(self):
        # Text field (second part) with the calculation results
        last_savings_rate = self.df.iloc[-2]['Monthly Savings']
        total_deposits = self.df['Monthly Savings'].sum() + self.parameters.initial_investment
        total_interest = self.df.iloc[-1]['Interest Payout']
        total_management_fees = self.df.iloc[-1]['Total Costs']
        final_assets = self.df.iloc[-1]['Month End']
        closing_costs = self.df.iloc[-1]['Costs']
        roi_before_tax = ((final_assets - total_deposits) / total_deposits) * 100
        tax = (final_assets - total_deposits) / 100 * 26.375
        roi_after_tax = ((final_assets - tax - total_deposits) / total_deposits) * 100

        text = r"$\bf{Ergebnisse}$"
        if self.parameters.dynamic_increase > 0:
            text += f"\nLetzte Sparrate: {last_savings_rate:.2f}€"
        if self.parameters.interest_volatility > 0:
            text += f"\nDurchschnittlicher Zinssatz (p.a.): {round(self.df['Interest Rate'].mean(), 2)}%"
        text += (
            f"\nSumme der Einzahlungen: {total_deposits:.2f}€"
            f"\nGesamter Zinsertrag: {total_interest:.2f}€"
            f"\nVerwaltungskosten (variabel+fix): {total_management_fees:.2f}€"
            f"\nSparvermögen zum Ende der Laufzeit: {final_assets + closing_costs:.2f}€"
            f"\nAbschlusskosten (variabel+fix): {closing_costs:.2f}€"
            f"\nGesamtkosten: {total_management_fees:.2f}€"
            f"\nEndkontostand nach Abschlusskosten: {final_assets:.2f}€"
            f"\nKapitalertragssteuer inkl. Solidaritätszuschlag (26.375%): {tax:.2f}€"
            f"\nROI vor Steuern: {roi_before_tax:.2f}%"
            f"\nEndvermögen nach Steuern: {final_assets - tax:.2f}€"
            f"\nROI nach Steuern: {roi_after_tax:.2f}%"
        )
        return text


# ------------------------------------------------- Program Execution -------------------------------------------------

class InvestmentOptimizer:
    def __init__(self):
        self.parameters = None
        self.calculator = None
        self.visualizer = None
        self.results = []

        self.param_names = {
            'initial_investment': 'Ersteinzahlung (€)',
            'duration': 'Laufzeit (Jahre)',
            'initial_monthly_savings': 'Anf. Sparrate (€)',
            'dynamic_increase': 'Dynamik (%)',
            'interest_rate': 'Zinssatz (%)',
            'interest_volatility': 'Zinsvolatilität (%)',
            'payout_interval': 'Zinsauszahlung (Monat/Quartal/Jahr)',
            'management_fee_variable': 'Verwaltungskosten (variabel) (%)',
            'management_fee_fixed': 'Verwaltungskosten (fix) (€)',
            'closing_costs_variable': 'Abschlusskosten (variabel) (%)',
            'closing_costs_fixed': 'Abschlusskosten (fix) (€)',
            'seed': 'Seed-Wert'
        }

        self.result_names = {
            'last_savings_rate': 'Höhe der letzten Sparrate (€)',
            'total_deposits': 'Summe der Einzahlungen über die Laufzeit (€)',
            'total_interest': 'Summe der Zinserträge über die Laufzeit (€)',
            'total_management_fees': 'Gesamtverwaltungskosten (variabel+fix) (€)',
            'final_value_before_closing_costs': 'Wert der Geldanlage zum Ende der Laufzeit (€)',
            'closing_costs': 'Abschlusskosten (variabel+fix) (€)',
            'final_value_after_closing_costs': 'Wert der Geldanlage zum Ende der Laufzeit (nach Abschlusskosten) (€)',
            'total_costs': 'Gesamtkosten (€)',
            'roi_before_tax': 'ROI (Return on Investment) der Geldanlage (%)',
            'tax': 'Kapitalertragssteuer inkl. Solidaritätszuschlag (26.375%) (€)',
            'final_value_after_tax': 'Endvermögen nach Steuern (€)',
            'roi_after_tax': 'ROI nach Steuern (%)'
        }

    def run(self):
        self._print_welcome_message()
        while True:
            self.parameters = self._get_user_input()
            self.calculator = InvestmentCalculator(self.parameters)
            df = self.calculator.calculate()
            self._print_results(df)
            self.visualizer = InvestmentVisualizer(df, self.parameters)
            fig = self.visualizer.visualize()

            result = {
                "parameters": self.parameters,
                "df": df,
                "figure": fig,
                "last_savings_rate": df.iloc[-2]['Monthly Savings'],
                "total_deposits": self.calculator.total_deposits,
                "total_interest": self.calculator.total_interest,
                "total_management_fees": self.calculator.total_management_fees,
                "final_value_before_closing_costs": df.iloc[-1]['Month End'] + df.iloc[-1]['Costs'],
                "closing_costs": df.iloc[-1]['Costs'],
                "final_value_after_closing_costs": df.iloc[-1]['Month End'],
                "total_costs": self.calculator.total_management_fees + df.iloc[-1]['Costs'],
                "roi_before_tax": ((df.iloc[-1][
                                        'Month End'] - self.calculator.total_deposits) / self.calculator.total_deposits) * 100,
                "tax": (df.iloc[-1]['Month End'] - self.calculator.total_deposits) * 0.26375,
            }
            result["final_value_after_tax"] = result["final_value_after_closing_costs"] - result["tax"]
            result["roi_after_tax"] = ((result[
                                            "final_value_after_tax"] - self.calculator.total_deposits) / self.calculator.total_deposits) * 100

            self.results.append(result)

            save_parameters(self.parameters, USER_PARAMS_FILE)

            if len(self.results) > 1:
                self._compare_results(self.results[-2], self.results[-1])

            another_run = input("\nMöchten Sie einen weiteren Durchlauf zum Vergleich starten? (j/n): ")
            if another_run.lower() != 'j':
                break

        print("\033[94m\nProgramm beendet. Vielen Dank für die Nutzung des Investment Optimizers!\033[0m")

    def _print_welcome_message(self):
        print("\033[94m========================\033[0m")
        print("\033[94m| Investment Optimizer |\033[0m")
        print("\033[94m========================\033[0m")

        print("\nDieses Programm berechnet die Entwicklung einer Geldanlage über einen bestimmten Zeitraum unter Berücksichtigung verschiedener Parameter.")
        print("    \033[32mAnlageparameter\033[0m: Anlagesumme, Laufzeit, Sparrate, Dynamik (jährliche Erhöhung der Sparrate), Zinssatz, Intervall der Zinsauszahlung")
        print("    \033[31mKostenparameter\033[0m: variable Verwaltungskosten (% der Sparrate), fixe Verwaltungskosten (von der Sparrate abgezogener Festbetrag)")
        print("                     fixe Abschlusskosten (% des Endvermögens), fixe Abschlusskosten (vom Endvermögen abgezogener Festbetrag)")
        print("\nNeben festverzinslichen Geldanlagen können auch volatile Erträge (wie etwa bei Aktienfonds oder ETFs) simuliert werden.")
        print("Die Ergebnisse werden sowohl tabellarisch als auch grafisch über die gesamte Anlagelaufzeit ausgegeben.")

    def _get_user_input(self):
        ### Prompts the user for investment parameters and returns an InvestmentParameters object
        ### Implements error messages for implausible inputs
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
                'initial_monthly_savings': 100,
                'dynamic_increase': 5,
                'interest_rate': 3.5,
                'interest_volatility': 0,
                'payout_interval': 1,
                'management_fee_variable': 1.5,
                'management_fee_fixed': 0,
                'closing_costs_variable': 0.5,
                'closing_costs_fixed': 0,
                'seed': None
            }

        initial_investment = self._get_float_input(f"Ersteinzahlung (in €) [Vorschlag: {params['initial_investment']}]: ",
            min_value=0, default=params['initial_investment'])
        duration = self._get_int_input(f"Laufzeit (in Jahren) [Vorschlag: {params['duration']}]: ",
                                       min_value=1, max_value=100, default=params['duration'])
        monthly_savings = self._get_float_input(f"Monatliche Sparrate (in €) [Vorschlag: {params['initial_monthly_savings']}]: ",
            min_value=0, default=params['initial_monthly_savings'])
        dynamic_increase = self._get_float_input(f"Jährliche Erhöhung der Sparrate (in %) [Vorschlag: {params['dynamic_increase']}]: ",
            min_value=0, max_value=100, default=params['dynamic_increase'])

        interest_type = self._get_int_input(f"Fester Zinssatz oder variabler Ertrag? '1'=fest, '2'=variabel [Vorschlag: {'2' if params['interest_volatility'] > 0 else '1'}]: ",
            min_value=1, max_value=2, default=2 if params['interest_volatility'] > 0 else 1
        )

        if interest_type == 1:
            interest_rate = self._get_float_input(f"Zinssatz (p.a.) [Vorschlag: {params['interest_rate']}]: ",
                                                  min_value=-100, max_value=100, default=params['interest_rate'])
            interest_volatility = 0
        else:
            interest_rate = self._get_float_input(f"Angenommener durchschnittlicher Zinssatz (p.a.) [Vorschlag: {params['interest_rate']}]: ",
                min_value=-100, max_value=100, default=params['interest_rate'])
            interest_volatility = self._get_float_input(f"Volatilität der Rendite (0-100) [Vorschlag: {params['interest_volatility'] or 10}]: ",
                min_value=0, max_value=100, default=params['interest_volatility'] or 10)

        seed = None
        if interest_type == 2:
            use_seed = self._get_int_input("Möchten Sie einen Seed-Wert für die Zufallszahlengenerierung verwenden? (1: Ja, 0: Nein) [Vorschlag: 1]: ",
                min_value=0, max_value=1, default=1)
            if use_seed:
                seed = self._get_int_input(f"Geben Sie den Seed-Wert ein (Ganzzahl) [Vorschlag: {params['seed'] or 42}]: ",
                    default=params['seed'] or 42)

        payout_interval = self._get_int_input(f"Zinsauszahlungsintervall (1: jährlich, 2: quartalsweise, 3: monatlich) [Vorschlag: {params['payout_interval']}]: ",
            min_value=1, max_value=3, default=params['payout_interval'])

        # Cost parameters
        print("\n\033[31mKostenparameter\033[0m:")
        print("----------------")
        management_fee_variable = self._get_float_input(f"Variable Verwaltungskosten (in % p.a.) [Vorschlag: {params['management_fee_variable']}]: ",
            min_value=0, max_value=100, default=params['management_fee_variable'])
        management_fee_fixed = self._get_float_input(f"Fixe Verwaltungskosten (in € p.a.) [Vorschlag: {params['management_fee_fixed']}]: ",
            min_value=0, default=params['management_fee_fixed'])
        closing_costs_variable = self._get_float_input(f"Variable Abschlusskosten (in %) [Vorschlag: {params['closing_costs_variable']}]: ",
            min_value=0, max_value=100, default=params['closing_costs_variable'])
        closing_costs_fixed = self._get_float_input(f"Fixe Abschlusskosten (in €) [Vorschlag: {params['closing_costs_fixed']}]: ",
            min_value=0, default=params['closing_costs_fixed'])

        print(f"\nFür die Berechnung der Verwaltungskosten wird von {management_fee_variable}% des Anlagebetrags sowie zusätzlich einem Fixbetrag von {management_fee_fixed}€ ausgegangen.")
        print(f"Zudem werden variable Abschlusskosten i.H.v. {closing_costs_variable}% vom Endvermögen und ggf. ein Fixbetrag von {closing_costs_fixed}€ abgezogen.")
        print(f"Außerdem wird die Kapitalertragssteuer i.H.v. 26.375% (inkl. Solidaritätszuschlag) berücksichtigt.")

        return InvestmentParameters(initial_investment, duration, monthly_savings, dynamic_increase, interest_rate,
                                    interest_volatility, payout_interval, management_fee_variable, management_fee_fixed,
                                    closing_costs_variable, closing_costs_fixed, seed)

    def _get_float_input(self, prompt, min_value=None, max_value=None, default=None):
        ### Plausibility check for float number inputs
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
        ### Plausibility check for integer number inputs
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

        # Asset development
        dynamic_increase = self.parameters.dynamic_increase
        total_deposits = self.calculator.total_deposits
        total_interest = self.calculator.total_interest

        if dynamic_increase > 0:
            last_savings_rate = df.iloc[-2]['Monthly Savings']
            print(f"Höhe der letzten Sparrate: \033[32m{last_savings_rate:.2f}€\033[0m")
        print(f"Summe der Einzahlungen über die Laufzeit: \033[32m{total_deposits:.2f}€\033[0m")
        print(f"Summe der Zinserträge über die Laufzeit: \033[32m{total_interest:.2f}€\033[0m")

        # KPIs: Returns vs. Costs
        total_management_fees = self.calculator.total_management_fees
        final_assets_after_closing_costs = df.iloc[-1]['Month End']
        closing_costs = df.iloc[-1]['Costs']
        print(f"\nGesamtverwaltungskosten (variabel+fix): \033[91m{total_management_fees:.2f}€\033[0m")
        print(f"Wert der Geldanlage zum Ende der Laufzeit: \033[32m{final_assets_after_closing_costs + closing_costs:.2f}€\033[0m")
        print(f"Abschlusskosten (variabel+fix): \033[91m{closing_costs:.2f}€\033[0m")
        print(f"Wert der Geldanlage zum Ende der Laufzeit (nach Abschlusskosten): \033[32m{final_assets_after_closing_costs:.2f}€\033[0m")
        print(f"Gesamtkosten: \033[91m{total_management_fees + closing_costs:.2f}€\033[0m")

        # Analysis metrics
        print(f"\nROI (Return on Investment) der Geldanlage: \033[32m{((final_assets_after_closing_costs - total_deposits) / total_deposits) * 100:.2f}%\033[0m")
        if self.parameters.interest_volatility > 0:
            print(f"Durchschnittlicher Zinssatz über die Laufzeit: \033[32m{df['Interest Rate'].iloc[1:-1].mean():.2f}%\033[0m")
            if self.parameters.seed is not None:
                print(f"Verwendeter Seed-Wert: \033[32m{self.parameters.seed}\033[0m")

        # Tax deduction
        tax = (final_assets_after_closing_costs - total_deposits) / 100 * 26.375
        print(f"\nKapitalertragssteuer inkl. Solidaritätszuschlag (26.375%): \033[31m{tax:.2f}€\033[0m")
        print(f"Endvermögen nach Steuern: \033[94m{final_assets_after_closing_costs - tax:.2f}€\033[0m")
        print(f"ROI nach Steuern: \033[94m{((final_assets_after_closing_costs - tax - total_deposits) / total_deposits) * 100:.2f}%\033[0m")


    # --------------------------------------------- Investment Comparison ---------------------------------------------

    def _compare_parameters(self, prev_params, curr_params):
        print("\nGegenüberstellung der Parameter:")
        print("--------------------------------")
        for key, value in vars(curr_params).items():
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
        # Parameter comparison
        param_table = []
        for key, value in vars(curr_result['parameters']).items():
            prev_value = getattr(prev_result['parameters'], key)
            param_name = self.param_names.get(key, key)
            if prev_value != value:
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
            'last_savings_rate', 'total_deposits', 'total_interest', 'total_management_fees',
            'final_value_before_closing_costs', 'closing_costs', 'final_value_after_closing_costs',
            'total_costs', 'roi_before_tax', 'tax', 'final_value_after_tax', 'roi_after_tax'
        ]

        for key in result_keys:
            prev_value = prev_result.get(key, 0)
            curr_value = curr_result.get(key, 0)
            if isinstance(prev_value, (int, float)) and isinstance(curr_value, (int, float)):
                diff = curr_value - prev_value
                result_table.append([self.result_names[key], f"{prev_value:.2f}", f"{curr_value:.2f}", f"{diff:+.2f}"])
            else:
                result_table.append([self.result_names[key], str(prev_value), str(curr_value), ""])

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
