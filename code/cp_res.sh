#!/bin/bash


echo 'copying results to results folder:' $1
echo 'test'
cp -r /home/jr0th/github/segmentation/logs /home/jr0th/github/segmentation/results/$1

echo 'copied logs'

cp -r ~/github/segmentation/out /home/jr0th/github/segmentation/results/$1
cp -r ~/github/segmentation/checkpoints /home/jr0th/github/segmentation/results/$1

echo 'done copying results.'
