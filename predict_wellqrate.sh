#!/bin/bash

# List the config files in conf/data/qsar
echo "Available QSAR configs:"
ls conf/data/qsar

# remove conf/data and replace / with . in the configfile name
for configfile in $(ls conf/data/qsar/*.yaml); do
  configfile=$(echo ${configfile} | sed 's|^conf/data/qsar/||')
  configfile=$(echo ${configfile} | sed 's|/|.|g')
  #strup .yaml
  configfile=$(echo ${configfile} | sed 's|.yaml||')
  echo "Running predictions with data=qsar/${configfile} ..."
  python -m eddde.predict data=qsar/${configfile} inference.batch_size=256 ddp=false experiment.name=${configfile}
done
