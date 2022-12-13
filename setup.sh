# check that data dir exists
if [ ! -d data ]; then
    echo "Data directory data does not exist"
    exit 1
fi
# create validation data
cd data || exit 1
unzip training.zip
unzip test_set_images.zip
mkdir -p validating/images
mkdir -p validating/groundtruth
for i in {81..100}
do
  mv training/images/satImage_$(printf "%03d" $i).png validating/images
  mv training/groundtruth/satImage_$(printf "%03d" $i).png validating/groundtruth
done
# create output dirs
mkdir -p logs
mkdir -p checkpoints
mkdir -p predictions
mkdir -p submissions