import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_testing_data():
    """
    Plots the testing data.
    """
    # sns.set_style("whitegrid")
    # sns.set_context("paper")

    # Load the data
    # data = sns.load_dataset("log/data_for_plotting/predict_val_0.csv")
    # data_mse = sns.load_dataset("log/data_for_plotting/predict_val_0.csv")

    data = pd.read_csv("log/data_for_plotting/predict_val_0.csv")
    data_mse = pd.read_csv("log/data_for_plotting/predict_val_0_mse.csv")
    # Plot the data
    # sns.relplot(x="date", y="signal", hue="class", data=data)
    plt.plot(data['real_q'][:53], label='real_q', color='blue')
    plt.plot(data['ensemble_q'][:53], label='linex', color='green')
    plt.plot(data_mse['ensemble_q'][:53], label='mse', color='red')
    
    plt.legend(loc='best')
    # Show the plot
    plt.savefig("log/data_for_plotting/predict.png")

def calculate_bias():
    data = pd.read_csv("log/data_for_plotting/predict_val_0.csv")
    data_mse = pd.read_csv("log/data_for_plotting/predict_val_0_mse.csv")

    diff = data['ensemble_q'].values - data['real_q'].values
    diff_mse = data_mse['ensemble_q'].values - data['real_q'].values

    bias = diff.sum() / diff.shape[0]
    bias_mse = diff_mse.sum() / diff_mse.shape[0]

    print(f"Bias: {bias}, bias mse: {bias_mse}")

if __name__ == "__main__":
    # plot_testing_data()
    calculate_bias()