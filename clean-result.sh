#!/bin/bash
for dir in omz-demo ov-api pot;
do
    cd $dir; ./clean-result.sh; cd ..
done
