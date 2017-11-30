#!/bin/bash

iter=15

# Basic while loop
counter=0
while [ $counter -le $iter ]
do
    printf "$counter: "
    ./out imgs/dart$counter.jpg imgs/out$counter.jpg
    ((counter++))
    printf "done.\n"
done
echo completed
