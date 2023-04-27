#!/bin/bash
cd lesson
for dir in `ls`;
do
    cd $dir; ./clean-result.sh; cd ..
done
cd ..
