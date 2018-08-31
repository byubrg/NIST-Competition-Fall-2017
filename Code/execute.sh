# /bin/bash

#Directory variables
redirectedTempFolder=tmp
softwareFolder=Software

#Downloaded and Installed in install.sh
minicondaPath=$softwareFolder/miniconda/bin/

# multilayer_preceptron output
MLPGenusOut=$redirectedTempFolder/gen.csv
MLPSpecisOut=$redirectedTempFolder/species.csv

# final output
finalGenusOut=genus_id_subm.csv
finalSpeciesOut=species_id_subm.csv

#Explorations
exploreFolder=futher_explorations
genusMLPExploration=$exploreFolder/mlp_genus.csv
speciesMLPExploration=$exploreFolder/mlp_species.csv
mergedMLPGenusExploration=$exploreFolder/merged_mlp_genus.csv
mergedMLPSpeciesExploration=$exploreFolder/merged_mlp_species.csv
genusRFExploration=$exploreFolder/RF_genus.csv
speciesRFExploration=$exploreFolder/RF_species.csv
mergedRFGenusExploration=$exploreFolder/merged_RF_genus.csv
mergedRFSpeciesExploration=$exploreFolder/merged_RF_species.csv
genusSVMExploration=$exploreFolder/SVM_genus.csv
speciesSVMExploration=$exploreFolder/SVM_species.csv
mergedSVMGenusExploration=$exploreFolder/merged_SVM_genus.csv
mergedSVMSpeciesExploration=$exploreFolder/merged_SVM_species.csv

mkdir -p $redirectedTempFolder

#miniconda environment is used to store skikitlearn_env
echo "Setting up environment"
#cd $minicondaPath
#source activate skikitLearn_env 
#cd ../../..

## Making predictions
#python classify_trees.py
#python mergePixels.py $MLPGenusOut $finalGenusOut "Genus"
#python mergePixels.py $MLPSpecisOut $finalSpeciesOut "Species"

# Futher explorations
python3 explore.py
# MLP
python3 mergePixels.py $genusMLPExploration $mergedMLPGenusExploration "Genus"
python3 mergePixels.py $speciesMLPExploration $mergedMLPSpeciesExploration "Species"
# RF
python3 mergePixels.py $genusRFExploration $mergedRFGenusExploration "Genus"
python3 mergePixels.py $speciesRFExploration $mergedRFSpeciesExploration "Species"
# SVM 
python3 mergePixels.py $genusSVMExploration $mergedSVMGenusExploration "Genus"
python3 mergePixels.py $speciesSVMExploration $mergedSVMSpeciesExploration "Species"
