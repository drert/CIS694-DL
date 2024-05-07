from PIL import Image
import argparse

# quick utility for PNG --> JPG conversion 

parser = argparse.ArgumentParser()
parser.add_argument('-i','--img_path', type=str)
opt = parser.parse_args()
file = opt.img_path
im = Image.open(file + ".png")
rgb_im = im.convert("RGB")
rgb_im.save(file + ".jpg")