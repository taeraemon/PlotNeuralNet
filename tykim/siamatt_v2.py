import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import *

# Architecture for the Siamese neural network
arch = [
    to_head('.'), 
    to_cor(),
    to_begin(),
    
    
    
    
    
    # ==== UpdateNet ==== #
    to_Conv(name="template_0", s_filer="", n_filer="", offset="(-8,0,0)", to="(0,0,0)", width=2, height=12, depth=12, caption="$T_{0}$"),
    to_Conv(name="template_i-1", s_filer="", n_filer="", offset="(-8,-4,0)", to="(0,0,0)", width=2, height=12, depth=12, caption="$T_{i-1}$"),
    to_Conv(name="template_i", s_filer="", n_filer="", offset="(-8,-8,0)", to="(0,0,0)", width=2, height=12, depth=12, caption="$T_{i}$"),
    
    to_Conv(name="updatenet", s_filer="", n_filer="", offset="(-5,0,0)", to="(0,0,0)", width=8, height=6, depth=6, caption="UpdateNet"),
    
    
    # ==== SiamAtt ==== #
    
    # ResNet50 blocks for both template and detection images
    to_Conv(name="resnet_template", s_filer="", n_filer="", offset="(-1,0,0)", to="(0,0,0)", width=2, height=12, depth=12, caption="ResNet50"),
    to_Conv(name="resnet_detection", s_filer="", n_filer="", offset="(-1,-8,0)", to="(0,0,0)", width=2, height=12, depth=12, caption="ResNet50"),
    
    # Convolution and Correlation layers (using Conv to simulate Corr layer)
    to_Conv(name="fm1_template", s_filer=" ", n_filer=" ", offset="(1,0,0)", to="(0,0,0)", width=4, height=8, depth=8, caption="7x7x256"),
    to_Conv(name="fm1_detection", s_filer=" ", n_filer=" ", offset="(1,-8,0)", to="(0,0,0)", width=4, height=8, depth=8, caption="31x31x256"),
    
    to_Conv(name="conv1_template", s_filer=" ", n_filer=" ", offset="(3,1.5,0)", to="(0,0,0)", width=2, height=4, depth=4, caption="Conv"),
    to_Conv(name="conv1_detection", s_filer=" ", n_filer=" ", offset="(3,-6.5,0)", to="(0,0,0)", width=2, height=4, depth=4, caption="Conv"),
    
    to_Conv(name="conv2_template", s_filer=" ", n_filer=" ", offset="(3,-1.5,0)", to="(0,0,0)", width=2, height=4, depth=4, caption="Conv"),
    to_Conv(name="conv2_detection", s_filer=" ", n_filer=" ", offset="(3,-9.5,0)", to="(0,0,0)", width=2, height=4, depth=4, caption="Conv"),
    
    
    to_Conv(name="fm11_template", s_filer=" ", n_filer=" ", offset="(5,1.5,0)", to="(0,0,0)", width=8, height=6, depth=6, caption="5x5x256"),
    to_Conv(name="fm11_detection", s_filer=" ", n_filer=" ", offset="(5,-6.5,0)", to="(0,0,0)", width=8, height=6, depth=6, caption="5x5x256"),
    
    to_Conv(name="fm12_template", s_filer="", n_filer="", offset="(5,-1.5,0)", to="(0,0,0)", width=8, height=6, depth=6, caption="5x5x256"),
    to_Conv(name="fm12_detection", s_filer="", n_filer="", offset="(5,-9.5,0)", to="(0,0,0)", width=8, height=6, depth=6, caption="5x5x256"),
    
    to_Conv(name="corr1", s_filer=" ", n_filer=" ", offset="(8,0,0)", to="(0,0,0)", width=2, height=4, depth=4, caption="Corr"),
    to_Conv(name="corr2", s_filer=" ", n_filer=" ", offset="(8,-8,0)", to="(0,0,0)", width=2, height=4, depth=4, caption="Corr"),
    
    to_Conv(name="fm2_template", s_filer="", n_filer="", offset="(10,0,0)", to="(0,0,0)", width=6, height=4, depth=4, caption="25x25x10"),
    to_Conv(name="fm2_detection", s_filer="", n_filer="", offset="(10,-8,0)", to="(0,0,0)", width=6, height=4, depth=4, caption="25x25x256"),
    
    # Additional layers and branches
    to_Conv(name="classification_conv", s_filer="", n_filer="", offset="(13,0,0)", to="(0,0,0)", width=6, height=6, depth=6, caption="Classification Branch"),
    to_Conv(name="attention_conv", s_filer="", n_filer="", offset="(13,-8,0)", to="(0,0,0)", width=6, height=6, depth=6, caption="Attention Branch"),
    
    to_Conv(name="fm3_template", s_filer="", n_filer="", offset="(15,0,0)", to="(0,0,0)", width=8, height=1, depth=1, caption="3125x1"),
    to_Conv(name="fm3_detection", s_filer="", n_filer="", offset="(15,-8,0)", to="(0,0,0)", width=8, height=1, depth=1, caption="3125x1"),
    
    # Concatenate and Weighted Sum (using Sum for weighted sum effect)
    to_Sum(name="weighted_sum", offset="(17,-4,0)", to="(0,0,0)", radius=2, opacity=0.6),

    # Final output layer (simulated with Conv as fully connected layer)
    to_Conv(name="output", s_filer="", n_filer="", offset="(18,-4,0)", to="(0,0,0)", width=8, height=1, depth=1, caption="Output"),




    # ==== UpdateNet ==== #
    to_connection("template_0", "updatenet"),
    to_connection("template_i-1", "updatenet"),
    to_connection("template_i", "updatenet"),
    
    to_connection("updatenet", "resnet_template"),
    to_connection("template_i", "resnet_detection"),

    # ==== SiamAtt ==== #

    to_connection("resnet_template", "fm1_template"),
    to_connection("resnet_detection", "fm1_detection"),
    
    to_connection("fm1_template", "conv1_template"),
    to_connection("fm1_detection", "conv2_template"),
    to_connection("fm1_template", "conv1_detection"),
    to_connection("fm1_detection", "conv2_detection"),
    
    to_connection("conv1_template", "fm11_template"),
    to_connection("conv1_detection", "fm11_detection"),
    to_connection("conv2_template", "fm12_template"),
    to_connection("conv2_detection", "fm12_detection"),
    
    
    to_connection("fm11_template", "corr1"),
    to_connection("fm12_template", "corr1"),
    to_connection("fm11_detection", "corr2"),
    to_connection("fm12_detection", "corr2"),
    
    to_connection("corr1", "fm2_template"),
    to_connection("corr2", "fm2_detection"),
    
    to_connection("fm2_template", "classification_conv"),
    to_connection("fm2_detection", "attention_conv"),
    
    
    to_connection("classification_conv", "fm3_template"),
    to_connection("attention_conv", "fm3_detection"),
    
    to_connection("fm3_template", "weighted_sum"),
    to_connection("fm3_detection", "weighted_sum"),

    to_connection("weighted_sum", "output"),
    
    to_end()
]

def main():
    namefile = 'siamatt_v2.tex'
    to_generate(arch, namefile)

if __name__ == '__main__':
    main()
