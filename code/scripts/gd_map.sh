#!/bin/bash

i=$1
i=$((i+1))
x=$(sed "${i}q;d" parameters.in | awk '{print $1}')
y=$(sed "${i}q;d" parameters.in | awk '{print $2}')
z=$(sed "${i}q;d" parameters.in | awk '{print $3}')
w=$(sed "${i}q;d" parameters.in | awk '{print $4}')

mkdir results
./CI $x $y $z $w 1000