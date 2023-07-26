#!/bin/bash
list=('python3 main.py --holding_only', ' python3 main.py --skipping_only')
for ((i=1; i<=100; i++))
do
    echo "Running iteration $i"
    list[i]
done