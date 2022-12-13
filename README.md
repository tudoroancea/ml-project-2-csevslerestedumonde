# Project Road Segmentation

## Models


## Setup instructions
On local computer
```bash
export IZAR_USERNAME="oancea"
scp -r data $IZAR_USERNAME@izar.epfl.ch:/home/$IZAR_USERNAME/road_segmentation/data
scp setup.sh $IZAR_USERNAME@izar.epfl.ch:/home/$IZAR_USERNAME/road_segmentation/
scp launch_jupyter.sh $IZAR_USERNAME@izar.epfl.ch:/home/$IZAR_USERNAME/road_segmentation/
```

On Izar
```bash
cd road_segmentation
./setup.sh
module load gcc python/3.7.7
python3 -m venv venv
source venv/bin/activate
pip3 install torch torchvision --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install -U pip albumentations matplotlib ipython jupyter tqdm
```