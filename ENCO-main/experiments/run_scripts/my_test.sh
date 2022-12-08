#!/bin/sh
# Graphs are dynamically generated here to save disk memory

cd ../  # Go back to experiment direction
# Graph with 100 variables
python run_exported_graphs.py --graph_files ../hihi.pt
