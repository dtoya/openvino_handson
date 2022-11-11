#!/bin/bash
for dir in omz-demo ov-api pot;
do
    cd $dir; ./test.sh; cd ..
done
