#!/bin/bash

# Basic while loop
counter=0
while [ $counter -le 16 ]
do
    ./out imgs/dart$counter.jpg imgs/out$counter.jpg
    ((counter++))
done
echo All done
