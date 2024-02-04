#!/bin/bash
for i in $(seq 1 100);
do
    pytest test_marlin.py
done