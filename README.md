# JugglingwithPandas
This repository is to show case some wonderful one-liners I made using Panda
The script go through large set of .root files re-arrange the data sets according to a choosen set of indices
The data set is public and can be read using CERNBox links. The data correspond to a research to pinpoint matter-antimatter asymmetries in nature (B0->JpsiK* decays at LHCb experiment)
Perparedata.py is a script which shows how to select entries in panda data frame based on a selection applied at the level of sub-index, this cleaning procedure include: simple selection applied on the feature and more complicated cleaning where the condition is applied on the sub-index given a condition on a feature (highest pT of the track ensemble).
The end goal of this excercise is to show how to turn a large/complicated .root file into a datframe which is read and understood by Keras models.
