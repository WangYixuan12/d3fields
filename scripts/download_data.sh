#!/bin/bash

gdown 1dt2CAcFq-sryF5gvrydXcWc9F1Jfd4I8 # mug.zip
gdown 1yGnMPaf8dtdnd0l3Z7vP1IpS3f1xAM_2 # fork.zip
gdown 1mglorNj158hs2buyfcKRZgTAl_JfiXl5 # shoe.zip
gdown 1b0YTwguqpCotjv1DtZLKFNObMva3IAtI # shoe_tracking.zip

mkdir -p data

unzip mug.zip -d data/
unzip fork.zip -d data/
unzip shoe.zip -d data/
unzip shoe_tracking.zip -d data/

rm mug.zip
rm fork.zip
rm shoe.zip
rm shoe_tracking.zip
