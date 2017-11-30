#!/bin/sh

#SBATCH --verbose
#SBATCH --job-name=jz2653-test
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=t_%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB

/bin/hostname
/bin/pwd

source .venv/bin/activate

python -u train.py ./images ./styles

# rembrandt_woman-standing-with-raised-hands.jpg
# vincent-van-gogh_l-arlesienne-portrait-of-madame-ginoux-1890.jpg

# train 56066
# test 16019
# val 8010
