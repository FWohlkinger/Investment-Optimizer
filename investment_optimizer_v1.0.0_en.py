##################################
###   Investment Optimizer     ###
##################################
### Author: Florian Wohlkinger ###
### Date: 08.06.2024           ###
### Version: 1.0.0             ###
##################################


# Importing modules
import pandas as pd                             # for Data Frame
import numpy as np                              # for normally distributed random variable
import matplotlib.pyplot as plt                 # for visualization
from matplotlib.ticker import MultipleLocator   # for visualization (--> Axis labeling)


# ----------------------------------------------- Investment Calculator Function -----------------------------------------------

def investment_calculator(initial_investment, runtime, monthly_savings, dynamics, interest_rate, interest_volatility, interval, variable_management_costs, fixed_management_costs, variable_closure_costs, fixed_closure_costs):
    data = []                                   # Creating dataset
    total_investment = initial_investment       # Stores the deposits made over time
    total_interest = 0                          # Stores the interest earned over time
    total_management_costs = 0                  # Stores the management costs incurred over time

    # Row 0 before the start of the runtime ("Month 0") --> Runtime begins with the initial investment
    data.append([0, 0, 0, 0, 0, 0, 0, 0, 0, initial_investment, 0])

    # Loop for each year
    for year in range(1, runtime + 1):
        annual_interest = 0                                 # Annual interest for each calendar year
        balance = initial_investment                        # Current balance as the basis for interest calculation
        interest_rate_previous_month = interest_rate        # Initializing the interest rate for the first month

        # Loop for each month
        for month in range(1, 13):
            interest_rate_month = interest_rate             # Helper variable (to not overwrite the original input)

            ### Procedure for volatile interest rates: Generating a normally distributed random variable
            #   dependent on the interest rate in the previous month --> autoregressive component (AR)
            interest_stddev = interest_rate * interest_volatility / 100
            random_variable = np.random.normal(0, interest_stddev)
            AR_parameter = 0.5          # AR parameter for the AR(1) process

            if interest_stddev == 0:    # Fixed interest rate (no volatility)
                interest_rate_month = interest_rate
            else:                       # Variable interest rate --> AR model
                interest_rate_month = interest_rate + AR_parameter * (interest_rate_previous_month - interest_rate) + random_variable
            interest_rate_previous_month = interest_rate_month  # Update interest rate for the next month

            # Calculating management costs and monthly investment rate
            management_costs_percentage = monthly_savings * variable_management_costs / 100
            management_costs = management_costs_percentage + fixed_management_costs
            investment_rate = monthly_savings - management_costs

            month_start = balance                               # Month start equals the last balance
            balance += investment_rate                          # Adding the monthly investment rate
            month_interest = balance * ((interest_rate_month / 100) / 12) # Calculating the interest earned in the month

            annual_interest += month_interest                   # Updating annual interest
            total_interest += month_interest                    # Updating total interest
            total_investment += monthly_savings                 # Updating total investment
            total_management_costs += management_costs          # Updating total management costs
            data.append([year, month, interest_rate_month, round(month_start, 2), monthly_savings, management_costs,
                         total_management_costs, investment_rate, round(month_interest, 2), round(balance, 2), 0])

            # Interest payout according to interval
            if interval == 1:      # yearly
                if month == 12:                               # Interest payout at the end of the year
                    data[-1][-1] = round(annual_interest, 2)
                    initial_investment = balance + data[-1][-1]
            elif interval == 2:    # quarterly
                if month % 3 == 0:                            # Interest payout at the end of each quarter
                    data[-1][-1] = round(annual_interest, 2)
                    initial_investment = balance + data[-1][-1]
                    annual_interest = 0                       # Reset annual interest ("quarterly interest") after payout
                    balance = initial_investment              # Balance set to the level after interest payout
            elif interval == 3:    # monthly
                data[-1][-1] = round(month_interest, 2)
                initial_investment = balance + data[-1][-1]   # Balance update is done monthly
                balance = initial_investment                  # Balance set to the level after interest payout

        # Possibly increasing the savings rate annually by the percentage of dynamics
        if dynamics > 0 and year < runtime:
            dynamics_amount = round(monthly_savings * (dynamics / 100), 2)
            monthly_savings = round(monthly_savings + dynamics_amount, 2)

    # Adding the last row at the end of the dataset --> includes the final balance including the last interest payout
    final_assets = balance + data[-1][-1]                                           # Final assets before deduction of closure costs
    variable_closure_costs = round(final_assets * variable_closure_costs / 100, 2)  # Calculating the variable closure costs
    closure_costs = variable_closure_costs + fixed_closure_costs                    # Sum of both closure cost types
    final_assets_after_closure_costs = final_assets - closure_costs                 # Final assets after deduction of closure costs
    data.append([runtime + 1, 0, 0, 0, 0, closure_costs, total_management_costs+closure_costs, 0, 0, final_assets_after_closure_costs, total_interest])

    # Creating dataset
    columns = ["Year", "Month", "Interest Rate", "Month Start", "Monthly Savings", "Costs", "Total Costs",
               "Monthly Investment Rate", "Month Interest", "Month End", "Interest Payout"]
    df = pd.DataFrame(data, columns=columns)

    return df, round(total_interest, 2), final_assets_after_closure_costs, total_investment, total_management_costs, closure_costs


# --------------------------------------------------- Program Call ---------------------------------------------------

### Entering parameters and calling the function
def main():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    pd.options.display.float_format = '{:.2f}'.format

    print("\033[94m========================\033[0m")
    print("\033[94m| Investment Optimizer |\033[0m")
    print("\033[94m========================\033[0m")

    print("\nThis program calculates the development of an investment over a certain period of time considering various parameters.")
    print("\n\033[32mInvestment Parameters\033[0m: Initial investment, Runtime, Monthly savings, Dynamics (annual increase in savings rate), Interest rate, Interest payout interval")
    print("\033[31mCost Parameters\033[0m: Variable management costs (% of monthly savings), Fixed management costs (fixed amount deducted from monthly savings)")
    print("                 Variable closure costs (% of final assets), Fixed closure costs (fixed amount deducted from final assets)")
    print("\nBesides fixed-income investments, volatile returns (as with equity funds or ETFs) can also be simulated. The results are presented both in tabular form and graphically over the entire investment period.")

    # Investment Parameters
    print("\n\033[32mInvestment Parameters\033[0m:")
    print("----------------------")
    initial_investment = float(input("Initial Investment Amount: "))
    runtime = int(input("Planned Runtime in Years: "))
    monthly_savings = float(input("Monthly Savings: "))
    dynamics = float(input("Annual Increase in Savings Rate (in percentage, '0' for no dynamics): "))
    interest_type = int(input("Fixed Interest Rate or Variable Return? '1'=fixed, '2'=variable: "))
    interest_rate = float(input("Interest Rate (per annum): ")) if interest_type == 1 else float(
        input("Assumed Average Interest Rate (per annum): "))
    interest_volatility = float(input("Volatility of Interest Rate (0-100): ")) if interest_type == 2 else 0
    interval = int(input("Interest Payout Interval: '1'=annually, '2'=quarterly, '3'=monthly: "))

    # Cost Parameters
    print("\n\033[31mCost Parameters\033[0m:")
    print("----------------")
    variable_management_costs = float(input("Variable Management Costs (as percentage of monthly savings): "))  # Percentage deduction from deposits
    fixed_management_costs = float(input("Fixed Management Costs (in € per month): "))                          # Fixed amount --> Deduction from monthly deposits
    variable_closure_costs = float(input("Variable Closure Costs (as percentage of final assets): "))           # Percentage deduction from final assets (before taxes)
    fixed_closure_costs = float(input("Fixed Closure Costs (as fixed amount in €): "))                          # Fixed amount --> Deduction from final assets (before taxes)

    print(f"\nFor the calculation of management costs, it is assumed that {variable_management_costs}% of the investment amount is deducted as well as an additional fixed amount of {fixed_management_costs}€.")
    print(f"Moreover, variable closure costs amounting to {variable_closure_costs}% of the final assets and, if applicable, a fixed amount of {fixed_closure_costs}€ are deducted.")
    print(f"In addition, capital gains tax of 26.375% is taken into account. (*)")
    print("    (*) This percentage represents the German capital gains tax including solidarity surcharge.")


    # ------------------------------------------------ Function Call ------------------------------------------------

    df, total_interest, final_assets_after_closure_costs, total_investment, total_management_costs, closure_costs = investment_calculator(
        initial_investment, runtime, monthly_savings, dynamics, interest_rate, interest_volatility, interval, variable_management_costs,
        fixed_management_costs, variable_closure_costs, fixed_closure_costs
    )


    # ----------------------------------- Investment Development (--> DataFrame) -----------------------------------

    print("\nInvestment Development Over Time:")
    print("-----------------------------------------------")
    print(df)


    # ------------------------------------------- Calculation Results -------------------------------------------

    print("\nCalculation Results:")
    print("--------------------------")

    # Asset development
    if dynamics > 0:
        last_savings_rate = df.iloc[-2]['Monthly Savings']
        print(f"Final Monthly Savings Rate: \033[32m{last_savings_rate:.2f}€\033[0m")
    print(f"Total Deposits: \033[32m{total_investment:.2f}€\033[0m")
    print(f"Total interest earned over the runtime: \033[32m{total_interest:.2f}€\033[0m")

    # Returns vs. Costs
    print(f"Total management costs (variable+fixed): \033[91m{total_management_costs:.2f}€\033[0m")
    print(f"Value of the investment at the end of the runtime: \033[32m{final_assets_after_closure_costs+closure_costs:.2f}€\033[0m")
    print(f"Closure costs (variable+fixed): \033[91m{closure_costs:.2f}€\033[0m")
    print(f"Value of the investment at the end of the runtime (after closure costs): \033[32m{final_assets_after_closure_costs:.2f}€\033[0m")
    print(f"Total costs: \033[91m{total_management_costs + closure_costs:.2f}€\033[0m")

    # Tax deduction
    tax = (final_assets_after_closure_costs - total_investment) / 100 * 26.375  # The percentage represents the German capital gains tax including solidarity surcharge.
    print(f"Capital gains tax (26.375%): \033[31m{tax:.2f}€\033[0m")
    print(f"Final assets after taxes: \033[94m{final_assets_after_closure_costs-tax:.2f}€\033[0m")

    # Analysis metrics
    print(f"\nROI (Return on Investment) of the investment: \033[32m{((total_interest-total_management_costs-closure_costs) / total_investment) * 100:.2f}%\033[0m")
    if interest_type == 2:
        print(f"Average interest rate over the runtime: \033[32m{round(df['Interest Rate'].iloc[1:-1].mean(), 2)}%\033[0m")


    # ------------------------------------------------- Visualization -------------------------------------------------

    df = df.iloc[1:].reset_index(drop=True)  # Deleting first row in the dataset and re-indexing
    plt.figure(figsize=(10, 6))

    # Line for balance over time
    # plt.plot(df.index, df['Month End'], label='Balance', color='blue')  # smooth curve
    plt.step(df.index, df['Month End'], where='post', label='Asset Status')  # plt.step --> "staircase"

    # Filling area under the balance line
    # plt.fill_between(df.index, df['Month End'], color='blue', alpha=0.3)                 # smooth curve
    plt.fill_between(df.index, df['Month End'], step='post', color='blue', alpha=0.3)  # step='post' -> "staircase"

    # Second line for cumulative deposits
    cumulative_deposits = df['Monthly Savings'].cumsum() + initial_investment
    # plt.plot(df.index, cumulative_deposits, label='Deposits', color='orange', linestyle='--') # smooth
    plt.step(df.index, cumulative_deposits, where='post', label='Deposits', linestyle='--') # staircase

    # Filling area under the line for cumulative deposits
    # plt.fill_between(df.index, cumulative_deposits, color='orange', alpha=0.3)                  # smooth
    plt.fill_between(df.index, cumulative_deposits, step='post', color='orange', alpha=0.3)  # staircase

    # Third line for cumulative costs over time
    plt.plot(df.index, df['Total Costs'], label='Costs', linestyle='--', color='red')

    # Filling area under the line for cumulative costs
    plt.fill_between(df.index, df['Total Costs'], step='post', color='red', alpha=0.3)  # staircase

    # Title and axis labels
    plt.title('Investment Development Over Time')
    plt.ylabel('Amount (in €)')
    plt.xlim(0, len(df) - 0.5)  # Adjusting x-axis limits

    # Different scaling of the x-axis depending on the runtime
    if runtime <= 4:
        plt.xlabel('Runtime (in months)')
        major_locator = MultipleLocator(3)  # Main interval: every quarter
        minor_locator = MultipleLocator(1)  # Sub-interval: every month
        plt.gca().xaxis.set_major_locator(major_locator)
        plt.gca().xaxis.set_minor_locator(minor_locator)
    elif runtime > 4 and runtime < 8:
        plt.xlabel('Runtime (in months)')
        major_locator = MultipleLocator(12)  # Main interval: every 12 months (1 year)
        minor_locator = MultipleLocator(3)  # Sub-interval: every 3 months
        plt.gca().xaxis.set_major_locator(major_locator)
        plt.gca().xaxis.set_minor_locator(minor_locator)
    else:
        plt.xlabel('Runtime (in years)')
        plt.xticks(np.arange(0, len(df), 12), labels=np.arange(0, len(df) // 12 + 1))

    # Legend text box with the entered investment parameters and the calculation results
    investment_text = (
            r"$\bf{Investment\ Parameters}$"
            + f"\nInitial Investment: {initial_investment:.2f}€"
            + f"\nRuntime: {runtime} Years"
            + f"\nInitial Monthly Savings: {monthly_savings:.2f}€"
            + f"\nDynamics: {dynamics:.2f}%"
            + f"\nManagement Costs (variable): {variable_management_costs}%"
            + f"\nManagement Costs (fixed): {fixed_management_costs}€"
            + f"\nClosure Costs (variable): {variable_closure_costs}%"
            + f"\nClosure Costs (fixed): {fixed_closure_costs}€"
            + f"\nAssumed Interest Rate (p.a.): {interest_rate}%"
            + f"\nInterest Payout: "
    )

    if interval == 1:
        investment_text += "annually"
    elif interval == 2:
        investment_text += "quarterly"
    elif interval == 3:
        investment_text += "monthly"

    results_text = r"$\bf{Calculation\ Results}$"
    if dynamics > 0:
        results_text += f"\nFinal Monthly Savings Rate: {last_savings_rate:.2f}€"
    if interest_type == 2:
        results_text += f"\nAverage Interest Rate (p.a.): {round(df['Interest Rate'].mean(), 2)}%"
    results_text += (
        f"\nTotal Deposits: {total_investment:.2f}€"
        f"\nTotal Interest Earned: {total_interest:.2f}€"
        f"\nManagement Costs (variable+fixed): {total_management_costs:.2f}€"
        f"\nInvestment Value at the End of the Runtime: {final_assets_after_closure_costs + closure_costs:.2f}€"
        f"\nClosure Costs (variable+fixed): {closure_costs:.2f}€"
        f"\nTotal Costs: {(total_management_costs + closure_costs):.2f}€"
        f"\nInvestment Value at the End of the Runtime (after closure costs): {final_assets_after_closure_costs:.2f}€"
        f"\nCapital gains tax (26.375%): {tax:.2f}€"
        f"\nFinal Assets after Taxes: {final_assets_after_closure_costs - tax:.2f}€"
    )

    plt.text(0.02, 0.98, investment_text + "\n\n" + results_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    # Show legend
    plt.legend(loc='lower right')

    # Show visualization
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------- Program Call -------------------------------------------------

if __name__ == "__main__":
    main()

