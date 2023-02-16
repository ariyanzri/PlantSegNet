python tune_hyperparameters.py -p /speedy/ariyanzarei/sorghum_segmentation/experiments/SorghumPartNetInstance/SPNS/EXP_08_SorghumPartNetInstance_SPNS.json
echo "---------------------------------------------------------------"
python tune_hyperparameters.py -p /speedy/ariyanzarei/sorghum_segmentation/experiments/SorghumPartNetInstance/SPNS/EXP_09_SorghumPartNetInstance_SPNS.json
echo "---------------------------------------------------------------"
python test.py -e /speedy/ariyanzarei/sorghum_segmentation/experiments/SorghumPartNetInstance/SPNS/EXP_08_SorghumPartNetInstance_SPNS.json
echo "---------------------------------------------------------------"
python test.py -e /speedy/ariyanzarei/sorghum_segmentation/experiments/SorghumPartNetInstance/SPNS/EXP_09_SorghumPartNetInstance_SPNS.json