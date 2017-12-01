#!/bin/bash

iter=15

rm -f f1scores
touch f1scores

counter=0
while [ $counter -le $iter ]
do
    printf "$counter: "
    ./out imgs/dart$counter.jpg -o imgs/out$counter.jpg -c cascade.xml
    ((counter++))
    printf "done.\n"
done
echo completed

total=$(cat f1scores | paste -sd+ | bc -l)
total=$(bc -l <<< $total/$iter)
echo "Average F1score = $total"
echo "Average F1score = $total" >> f1scores
