#!/bin/bash 

for eps in 0.5 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001 0.00005 0.00001 
do
echo $eps
python ../parm-opt.py pm3.txt $eps > output-$eps

cp fort.14 params.dat-$eps

cp energies.txt energies.txt-$eps


done
