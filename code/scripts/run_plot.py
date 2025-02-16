import argparse, sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis import plot_mapping, plot_gd, plotMapDiff, ploterror, partition  # Import only the functions you need

def main():
    parser = argparse.ArgumentParser(description="Run specific plotting functions from plotting.py")
    parser.add_argument("plot_function", type=str, help="Name of the plotting function to run")

    args = parser.parse_args()
    
    # Dictionary to map function names to actual functions
    plot_functions = {
        "plot_mapping": plot_mapping,
        "plot_gd": plot_gd,
        "partition": partition,
        "ploterror": ploterror,
        "plotMapDiff": plotMapDiff
    }

    if args.plot_function in plot_functions:
        print(f"Running {args.plot_function}...")
        plot_functions[args.plot_function]()
    else:
        print(f"Error: {args.plot_function} is not a valid plotting function.")

if __name__ == "__main__":
    main()