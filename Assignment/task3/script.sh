#!/bin/bash

iter=13

# Basic while loop
counter=0
while [ $counter -le $iter ]
do
    printf "$counter: "
    ./out imgs/dart$counter.jpg imgs/out$counter.jpg cascade.xml
    ((counter++))
    printf "done.\n"
done
echo completed
