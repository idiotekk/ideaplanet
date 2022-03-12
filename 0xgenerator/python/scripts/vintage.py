import sys, os
import pprint
import numpy as np
from PIL import ImageFilter
sys.path.insert(0, os.path.expandvars("$GITHUB/ideaplanet/0xgenerator/python"))
pprint.pprint(sys.path)
from py0xlab import *
from py0xlab import io
from neuralstyletransfer.style_transfer import NeuralStyleTransfer

nst = NeuralStyleTransfer()

input_dir = Path("/Users/zche/cloud/data/0xgenerator/andy/inputs/")
output_dir = Path("/Users/zche/cloud/data/0xgenerator/andy/outputs/") 


content_url = 'https://i.ibb.co/6mVpxGW/content.png'
style_url = 'https://i.ibb.co/30nz9Lc/style.jpg'
nst.LoadContentImage(content_url, pathType='url')
nst.LoadStyleImage(style_url, pathType='url')

output = nst.apply(contentWeight=1000, styleWeight=0.01, epochs=600)

io.save_im(output, output_dir / "nst_test.png")

