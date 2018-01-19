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

mkdir -p $redirectedTempFolder

#miniconda environment is used to store skikitlearn_env
echo "Setting up environment"
cd $minicondaPath
source activate skikitLearn_env 
cd ../../..
python classify_trees.py
python mergePixels.py $MLPGenusOut $finalGenusOut "Genus"
python mergePixels.py $MLPSpecisOut $finalSpeciesOut "Species"
