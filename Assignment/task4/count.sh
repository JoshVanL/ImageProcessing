#!/bin/bash

total=$(cat prec | paste -sd+ | bc -l)
total=$(bc -l <<< $total/15)
echo "Average precision = $total"
#echo "Average F1score = $total" >> f1scores
