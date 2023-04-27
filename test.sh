#!/bin/bash
cd lesson
for dir in `ls`;
do
    cd $dir; ./test.sh; cd ..
done
cd ..
