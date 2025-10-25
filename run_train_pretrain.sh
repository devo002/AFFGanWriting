#!/bin/bash

if [ -n "$1" ]; then
  ID=$1
else
  LAST_EPOCH=$(find /home/vault/iwi5/iwi5333h/save_weights/ -name contran-*.model |
      sed "s/.*contran-\([0-9]*\)\.model/\1/" |
      #sed "s/save\_weights\/contran\-\([0-9][0-9]*\)\.model/\1/" |
      sort -n |
      tail -n 1)
  echo "Starting training at epoch $LAST_EPOCH"
  ID=$LAST_EPOCH
fi

export CUDA_VISIBLE_DEVICES=0
python3 /home/woody/iwi5/iwi5333h/AFFGanWriting/main_runbad22.py $ID
