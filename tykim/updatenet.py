import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import *

# UpdateResNet256 네트워크 구조
arch = [
    to_head('..'), 
    to_cor(),
    to_begin(),
    
    # UpdateResNet256 Layers
    to_Conv(name="conv1", s_filer=768, n_filer=96, offset="(0,0,0)", to="(0,0,0)", width=5, height=32, depth=32, caption="Conv2d 768 $\\rightarrow$ 96"),
    to_Conv(name="relu1", s_filer=96, n_filer=96, offset="(0.5,0,0)", to="(conv1-east)", width=1, height=32, depth=32, caption="ReLU"),
    to_Conv(name="conv2", s_filer=96, n_filer=256, offset="(3,0,0)", to="(relu1-east)", width=4, height=16, depth=16, caption="Conv2d 96 $\\rightarrow$ 256"),
    
    
    to_connection("conv1", "relu1"),
    to_connection("relu1", "conv2"),
    
    to_end()
]

def main():
    namefile = 'updatenet.tex'
    to_generate(arch, namefile)

if __name__ == '__main__':
    main()
