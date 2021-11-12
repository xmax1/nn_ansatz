#!/bin/bash

source ~/.bashrc
screen -dm exp 
screen -p 0 -X stuff "$1^M"
