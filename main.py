import sys
import os
new_directory = os.path.abspath("QuantArt")
print(new_directory)
sys.path.insert(0, new_directory)
from QuantArt.generate_art import generate_landscape_art





ckpt_path = 'QuantArt/logs/landscape2art/checkpoints/last.ckpt'
images_paths = ['QuantArt/datasets/lhq_1024_jpg/lhq_1024_jpg\\download - Copy.jpg', 'QuantArt/datasets/lhq_1024_jpg/lhq_1024_jpg\\2560x1440-best-nature-4k_1540131754.jpg']
styles_paths =  ['QuantArt/datasets/painter-by-numbers/train\\1.jpg', 'QuantArt/datasets/painter-by-numbers/train\\4.jpg']
out_path = 'results'
generate_landscape_art(images_paths,styles_paths,ckpt_path,out_path)