#!/bin/sh

#SBATCH --verbose
#SBATCH --job-name=jz2653-test
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=t_%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --mail-type=END
#SBATCH --mail-user=jiadong.zhu@nyu.edu

/bin/hostname
/bin/pwd

source .venv/bin/activate

python -u train.py --content_folder=/scratch/jz2653/coco --style_folder=/scratch/jz2653/wikiart

# rembrandt_woman-standing-with-raised-hands.jpg
# vincent-van-gogh_l-arlesienne-portrait-of-madame-ginoux-1890.jpg

# train 56066
# test 16019
# val 8010
