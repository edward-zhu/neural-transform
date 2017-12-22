# Neural Style Transfer

## Train

    python train.py \
        --epoch=10 \
        --batch_size=8 \
        --content_folder=./path/to/content/images \
        --style_folder=./path/to/style/images

## Generate

    python test.py \
        --content_folder=./path/to/content/images \
        --style_folder=./path/to/style/images \
        --model_decoder=./path/to/decoder_*.model \
        --alpha=1.0 # control the degree of stylization