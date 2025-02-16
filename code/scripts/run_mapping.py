import argparse, sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis import grid_mapping, gradient_mapping, hap_mapping, hap_gradient_mapping
from utils import save_pickle

def main():
    parser = argparse.ArgumentParser(description="Run specific mapping functions from mapping.py")
    parser.add_argument("map_function", type=str, help="Name of the mapping function to run")

    args = parser.parse_args()
    
    # Dictionary to map function names to actual functions
    map_functions = {
        "grid": grid_mapping,
        "gradient": gradient_mapping,
        "hap_grid": hap_mapping,
        "hap_gradient": hap_gradient_mapping
    }

    if args.map_function in map_functions:
        print(f"Running {args.map_function}...")
        if args.map_function in {'hap_grid', 'hap_gradient'}:
            params = {'n': 500, 'h': 0, 'target_steps': 40000, 'q0': 0.001}
            mapping_result = map_functions[args.map_function](params)
        else:
            mapping_result = map_functions[args.map_function]()
        print('Running and saving the mapping results...')

        save_pickle(f"{args.map_function}_gametic_fix.pickle", mapping_result)
    else:
        print(f"Error: {args.map_function} is not a valid mapping function.")

if __name__ == "__main__":
    main()