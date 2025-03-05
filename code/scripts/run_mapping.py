import argparse, sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis import grid_mapping, gradient_mapping, hap_grid_mapping, hap_gradient_mapping, getdiff
from utils import save_pickle

def main():
    parser = argparse.ArgumentParser(description="Run specific mapping functions from mapping.py")
    parser.add_argument("map_function", type=str, help="Name of the mapping function to run")
    parser.add_argument("h", type=str, help="Value of H in GD")
    parser.add_argument("gdFile", type=str, help="Name of GDres file")

    args = parser.parse_args()
    
    # Dictionary to map function names to actual functions
    map_functions = {
        "grid": grid_mapping,
        "gradient": gradient_mapping,
        "hap_grid": hap_grid_mapping,
        "hap_gradient": hap_gradient_mapping
    }

    if args.map_function in map_functions:
        # print(f"Running {args.map_function}...")
        # if args.map_function in map_functions:
        #     params = {'n': 500, 'h': args.h, 'target_steps': 40000, 'q0': 0.001}
        #     mapping_result = map_functions[args.map_function](params, args.gdFile)
        # else:
        #     print('NOT IN MAP_FUNCTIONS', args.map_function)
        #     mapping_result = map_functions[args.map_function]()
        # print('Running and saving the mapping results...')

        # is001 = '001' if '001' in args.gdFile else ''

        # save_pickle(f"h{args.h}_{args.map_function}{is001}_G_fix.pickle", mapping_result)

        getdiff(args.h, args.map_function, args.gdFile)
    else:
        print(f"Error: {args.map_function} is not a valid mapping function.")

if __name__ == "__main__":
    main()