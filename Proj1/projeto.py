import matplotlib.pyplot as plt

import matplotlib.colors as clr  # color map

import numpy as np

import scipy.fftpack as fft

import math

image_names = ["barn_mountains", "logo", "peppers"]
images_path = "./imagens/"
imgs = []
color_map_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
color_map_colors_name = ["RED", "GREEN", "BLUE", "GRAY"]

Tc = np.array([[0.299, 0.587, 0.114],
               [-0.168736, -0.331264, 0.5],
               [0.5, -0.418688, -0.081312]])

QY = np.array([[16, 11, 10, 16,  24,  40,  51,  61],
               [12, 12, 14, 19,  26,  58,  60,  55],
               [14, 13, 16, 24,  40,  57,  69,  56],
               [14, 17, 22, 29,  51,  87,  80,  62],
               [18, 22, 37, 56,  68, 109, 103,  77],
               [24, 35, 55, 64,  81, 104, 113,  92],
               [49, 64, 78, 87, 103, 121, 120, 101],
               [72, 92, 95, 98, 112, 100, 103,  99]])

QC = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
               [18, 21, 26, 66, 99, 99, 99, 99],
               [24, 26, 56, 99, 99, 99, 99, 99],
               [47, 66, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99]])

yarray=[]

array2=[]



def load_images(images_extension):
    global imgs
    i = 0
    temp_imgs = []
    for name in image_names:
        name = images_path + name + "." + images_extension
        temp_imgs.insert(i, plt.imread(name))
        i += 1
    imgs = temp_imgs[:]  # copy to global var


def show_image(image, title):
    plt.figure()
    plt.title(title)
    if isinstance(image, list):
        plt.imshow(image[0], image[1])
    else:
        plt.imshow(image)


def get_rgb_components(img):
    RED = img[:, :, 0]
    GREEN = img[:, :, 1]
    BLUE = img[:, :, 2]

    return RED, GREEN, BLUE

def join_rgb_components(r,g,b):
    nl,nc=r.shape
    rgb = np.zeros([nl,nc,3])
    rgb[:,:,0] = r;
    rgb[:,:,1] = g;
    rgb[:,:,2] = b;

    return rgb


def reverse_get_rgb_components(RED, GREEN, BLUE):
    reversedImage = np.dstack((RED, GREEN, BLUE))
    return reversedImage


def color_map(color, title):
    color_map = clr.LinearSegmentedColormap.from_list(
        title, [(0, 0, 0), color_map_colors[color_map_colors_name.index(color)]], N=256
    )
    return color_map


def visulize_rgb_color_maps(image, title):
    plt.figure()
    for i in range(0, 3):
        color_map = clr.LinearSegmentedColormap.from_list(
            title, [(0, 0, 0), color_map_colors[i]], N=256
        )
        img = image[:, :, i]
        plt.subplot(1, 3, i + 1)
        plt.imshow(img, color_map)


def pading(p1,p2,p3):
    r, c = p1.shape
    repeat_r=0
    repeat_c=0
    if (r % 16) != 0:
        repeat_r=16-r%16
        
        lines = np.zeros([1,c])
        lines[0,:]=p1[-1, :]
        lines=np.repeat(lines,repeat_r,axis=0)
        p1=np.vstack([p1, lines])
        
        lines = np.zeros([1,c])
        lines[0,:]=p2[-1, :]
        lines=np.repeat(lines,repeat_r,axis=0)
        p2=np.vstack([p2, lines])
        
        lines = np.zeros([1,c])
        lines[0,:]=p3[-1, :]
        lines=np.repeat(lines,repeat_r,axis=0)
        p3=np.vstack([p3, lines])
        
    r, c = p1.shape
    if (c % 16) != 0:
        repeat_c=16-c%16
        
        column = np.zeros([r,1])
        column[:,0]=p1[:, -1]
        column=np.repeat(column,repeat_c,axis=1)
        p1 = np.column_stack([p1, column])
        
        column = np.zeros([r,1])
        column[:,0]=p2[:, -1]
        column=np.repeat(column,repeat_c,axis=1)
        p2 = np.column_stack([p2, column])
        
        column = np.zeros([r,1])
        column[:,0]=p3[:, -1]
        column=np.repeat(column,repeat_c,axis=1)
        p3 = np.column_stack([p3, column])
    
    return p1,p2,p3


def inverse_pading(r,g,b, nl, nc):
    return r[:nl, :nc],g[:nl, :nc],b[:nl, :nc]


def rgb2ycbcr(r,g,b):
    y = np.zeros(r.shape)
    cb = np.zeros(r.shape)
    cr = np.zeros(r.shape)

    # Y
    y = Tc[0][0] * r + Tc[0][1] * g + Tc[0][2] * b
    # Cb
    cb = 128 + Tc[1][0] * r + Tc[1][1] * g + Tc[1][2] * b
    # Cr
    cr = 128 + Tc[2][0] * r + Tc[2][1] * g + Tc[2][2] * b
    return y,cb,cr


def ycbcr2rgb(y,cb,cr):
    TcInvertida = np.linalg.inv(Tc)
    nl,nc=y.shape
    rgb = np.zeros([nl,nc,3])
    cb = cb - 128
    cr = cr - 128

    # R
    rgb[:, :, 0] = TcInvertida[0][0] * y + \
        TcInvertida[0][1] * cb + TcInvertida[0][2] * cr
    # G
    rgb[:, :, 1] = TcInvertida[1][0] * y + \
        TcInvertida[1][1] * cb + TcInvertida[1][2] * cr
    # B
    rgb[:, :, 2] = TcInvertida[2][0] * y + \
        TcInvertida[2][1] * cb + TcInvertida[2][2] * cr

    rgb = np.round(rgb)
    rgb[rgb > 255] = 255
    rgb[rgb < 0] = 0
    rgb=np.uint8(rgb)

    return rgb[:, :, 0],rgb[:, :, 1],rgb[:, :, 2]


def sub_amostragem(y,cb,cr,flag):
    scaleX = 0.5
    scaleY = 1
    #if flag ==1 then it is sub amostragem 4-2-0 else 4-2-2
    if(flag==1):
        scaleY = 0.5
    
    stepX = int(1//scaleX)
    stepY = int(1//scaleY)
    
    cb = cb[::stepY, ::stepX]
    cr = cr[::stepY, ::stepX]
    
    return y,cb,cr

def upsampling(y,cb,cr,flag):
    scaleX = 0.5
    scaleY = 1
    #if flag ==1 then it is sub amostragem 4-2-0 else 4-2-2
    if(flag==1):
        scaleY = 0.5
    
    stepX = int(1//scaleX)
    stepY = int(1//scaleY)
    
    cb = np.repeat(cb, stepX, axis=1)
    cb = np.repeat(cb, stepY, axis=0)
    cr = np.repeat(cr, stepX, axis=1)
    cr = np.repeat(cr, stepY, axis=0)
    
    return y,cb,cr
    
def show_ycbcr(y,cb,cr):
    plt.figure()
    gray_color_map = color_map("GRAY", "GRAY_COLOR_MAP")
    
    plt.subplot(1, 3, 1)
    plt.title("y")
    plt.imshow(y, gray_color_map)
    
    plt.subplot(1, 3, 2)
    plt.title("cb")
    plt.imshow(cb, gray_color_map)
    
    plt.subplot(1, 3, 3)
    plt.title("cr")
    plt.imshow(cr, gray_color_map)
    
def dct_encoder(y):
    y=fft.dct(fft.dct(y, norm="ortho").T, norm="ortho").T
    return y

def dct_decoder(y):
    y=fft.idct(fft.idct(y, norm="ortho").T, norm="ortho").T
    return y

def dct_all(y,cb,cr):
    return dct_encoder(y),dct_encoder(cb),dct_encoder(cr)

def dct_all_invert(y,cb,cr):
    return dct_decoder(y),dct_decoder(cb),dct_decoder(cr)

def dct_block(y,cb,cr,d):
    nl,nc=y.shape
    nl1,nc1=cb.shape
    for x in range(0,int(nl/d)):
        for c in range(0,int(nc/d)):
            posX=d*x
            posY=d*c
            y[posX:posX+d,posY:posY+d]=dct_encoder(y[posX:posX+d,posY:posY+d])
            
    for x in range(0,int(nl1/d)):
        for c in range(0,int(nc1/d)):
            posX=d*x
            posY=d*c
            cb[posX:posX+d,posY:posY+d]=dct_encoder(cb[posX:posX+d,posY:posY+d])
            cr[posX:posX+d,posY:posY+d]=dct_encoder(cr[posX:posX+d,posY:posY+d])
    
    
    return y,cb,cr

def dct_block_decoder(y,cb,cr,d):
    nl,nc=y.shape
    nl1,nc1=cb.shape
    for x in range(0,int(nl/d)):
        for c in range(0,int(nc/d)):
            posX=d*x
            posY=d*c
            y[posX:posX+d,posY:posY+d]=dct_decoder(y[posX:posX+d,posY:posY+d])
            
    for x in range(0,int(nl1/d)):
        for c in range(0,int(nc1/d)):
            posX=d*x
            posY=d*c
            cb[posX:posX+d,posY:posY+d]=dct_decoder(cb[posX:posX+d,posY:posY+d])
            cr[posX:posX+d,posY:posY+d]=dct_decoder(cr[posX:posX+d,posY:posY+d])
    
    
    return y,cb,cr

def quantizacao(y,cb,cr,q):
    nl,nc=y.shape
    nl1,nc1=cb.shape
    global QY
    global QC
    
    if(q==100):
        return np.round(y),np.round(cb),np.round(cr)
    if(q>=50):
        QY_ENC=np.round(QY*((100-q)/50))
        QC_ENC=np.round(QC*((100-q)/50))
    else:
        QY_ENC=np.round(QY*(50/q))
        QC_ENC=np.round(QC*(50/q))
        
    for x in range(0,int(nl/8)):
        for c in range(0,int(nc/8)):
            posX=8*x
            posY=8*c
            y[posX:posX+8,posY:posY+8]=np.round(y[posX:posX+8,posY:posY+8]/QY_ENC)
            
    for x in range(0,int(nl1/8)):
        for c in range(0,int(nc1/8)):
            posX=8*x
            posY=8*c
            cb[posX:posX+8,posY:posY+8]=np.round(cb[posX:posX+8,posY:posY+8]/QC_ENC)
            cr[posX:posX+8,posY:posY+8]=np.round(cr[posX:posX+8,posY:posY+8]/QC_ENC)
            
    return y,cb,cr
        
    
def inv_quantizacao(y,cb,cr,q):
    nl,nc=y.shape
    nl1,nc1=cb.shape
    
    global QY
    global QC
    
    if(q==100):
        return y,cb,cr
    if(q>=50):
        QY_DEC=np.round(QY*((100-q)/50))
        QC_DEC=np.round(QC*((100-q)/50))
    else:
        QY_DEC=np.round(QY*(50/q))
        QC_DEC=np.round(QC*(50/q))
        
    for x in range(0,int(nl/8)):
        for c in range(0,int(nc/8)):
            posX=8*x
            posY=8*c
            y[posX:posX+8,posY:posY+8]=np.round(y[posX:posX+8,posY:posY+8]*QY_DEC)
            
    for x in range(0,int(nl1/8)):
        for c in range(0,int(nc1/8)):
            posX=8*x
            posY=8*c
            cb[posX:posX+8,posY:posY+8]=np.round(cb[posX:posX+8,posY:posY+8]*QC_DEC)
            cr[posX:posX+8,posY:posY+8]=np.round(cr[posX:posX+8,posY:posY+8]*QC_DEC)
            
    return y,cb,cr

def dpcm(y,cb,cr,block):
    nl,nc=y.shape
    nl1,nc1=cb.shape
    dc0 = 0
    dc1 = 0
    dc2 = 0
    for i in range(0, nl, 8):
        for j in range(0, nc, 8):
            if i == 0 and j == 0:
                dc0 = y[i, j]
                continue
            dc=y[i, j]
            y[i, j]=dc-dc0
            dc0=dc
            
    for i in range(0, nl1, 8):
        for j in range(0, nc1, 8):
            if i == 0 and j == 0:
                dc1 = cb[i, j]
                dc2 = cr[i, j]
                continue
            dc=cb[i, j]
            cb[i, j]=dc-dc1
            dc1=dc
            
            dc=cr[i, j]
            cr[i, j]=dc-dc2
            dc2=dc
    return y,cb,cr

def inv_dpcm(y,cb,cr,block):
    nl,nc=y.shape
    nl1,nc1=cb.shape
    dc0 = 0
    dc1 = 0
    dc2 = 0
    nl1,nc1=cb.shape
    dc0 = 0
    dc1 = 0
    dc2 = 0
    for i in range(0, nl, 8):
        for j in range(0, nc, 8):
            if i == 0 and j == 0:
                dc0 = y[i, j]
                continue
            y[i, j]=y[i, j]+dc0
            dc0=y[i, j]
            
    for i in range(0, nl1, 8):
        for j in range(0, nc1, 8):
            if i == 0 and j == 0:
                dc1 = cb[i, j]
                dc2 = cr[i, j]
                continue
            cb[i, j]=cb[i, j]+dc1
            dc1=cb[i, j]
            
            cr[i, j]=cr[i, j]+dc2
            dc2=cr[i, j]
            
    return y,cb,cr
    
        
def show_dct(y,cb,cr):
    plt.figure()
    gray_color_map = color_map("GRAY", "GRAY_COLOR_MAP")

    plt.subplot(1, 3, 1)
    plt.title("y")
    plt.imshow(np.log(abs(y) + 0.0001), gray_color_map)
    
    plt.subplot(1, 3, 2)
    plt.title("cb")
    plt.imshow(np.log(abs(cb) + 0.0001), gray_color_map)
    
    plt.subplot(1, 3, 3)
    plt.title("cr")
    plt.imshow(np.log(abs(cr) + 0.0001), gray_color_map)
    
def show_dct2(y,cb,cr):
    plt.figure()
    gray_color_map = color_map("GRAY", "GRAY_COLOR_MAP")

    plt.subplot(1, 3, 1)
    plt.title("y")
    plt.imshow(y, gray_color_map)
    
    plt.subplot(1, 3, 2)
    plt.title("cb")
    plt.imshow(cb, gray_color_map)
    
    plt.subplot(1, 3, 3)
    plt.title("cr")
    plt.imshow(cr, gray_color_map)
    
def distortion_calc(original,decoded):
    nl,nc,_=original.shape
    original_float=original.astype(np.float32)
    decoded_float=decoded.astype(np.float32)
    mse = (1/(nl*nc))*np.sum((original_float - decoded_float) ** 2)
    print("MSE= ",mse)
    rmse= mse**(1/2)
    print("RMSE= ", rmse)
    p = (1/(nl*nc))*np.sum(original_float** 2)
    snr=10* math.log10(p/mse)
    print("SNR= ", snr)
    temp=np.max(original_float)
    psnr=10* math.log10((temp**2)/mse)
    print("PSNR= ", psnr)
    

def getydiff():
    plt.figure()
    gray_color_map = color_map("GRAY", "GRAY_COLOR_MAP")
    plt.subplot(1, 3, 1)
    plt.title("y_original")
    plt.imshow(yarray, gray_color_map)
    
    plt.subplot(1, 3, 2)
    plt.title("y_after_process")
    plt.imshow(array2, gray_color_map)
    
    plt.subplot(1, 3, 3)
    plt.title("y_reconstructed")
    plt.imshow(yarray-array2, gray_color_map)

def encoder(img):
    #3.2
    red_color_map = color_map("RED", "RED_COLOR_MAP")
    
    #3.3
    show_image([img[:, :, 0], red_color_map], "Color map image")
    
    #3.4
    r, g, b = get_rgb_components(img)
    
    #3.5
    visulize_rgb_color_maps(img, "all_rgb_maps")
    
    #4.1
    r,g,b = pading(r,g,b)
    #5.2
    y,cb,cr = rgb2ycbcr(r,g,b)
    
    #5.3
    show_ycbcr(y,cb,cr)
    
    #6.1
    y_d,cb_d,cr_d=sub_amostragem(y,cb,cr,1)
    
    #6.1
    show_ycbcr(y_d,cb_d,cr_d)
    #7.1.1
    
    global yarray
    yarray=np.array(y_d)
    
    Y_dct, Cb_dct, Cr_dct=dct_all(y_d,cb_d,cr_d)
    show_dct(Y_dct, Cb_dct, Cr_dct)
    
    y_dct8, cb_dct8, cr_dct8=dct_block(y_d,cb_d,cr_d,8)
    
    #7.1.2
    #figura 6
    show_dct(y_dct8, cb_dct8, cr_dct8)
    
    y_quant,cb_quant,cr_quant=quantizacao(y_dct8, cb_dct8, cr_dct8,10)
    #figura 7
    show_dct(y_quant,cb_quant,cr_quant)

    y_dpcm,cb_dpcm,cr_dpcm=dpcm(y_quant,cb_quant,cr_quant,8)

    #figura 8
    show_dct(y_dpcm,cb_dpcm,cr_dpcm)
    
    return y_dpcm,cb_dpcm,cr_dpcm

def decoder(Y_dct, Cb_dct, Cr_dct, nl, nc):
    
    Y_dct, Cb_dct, Cr_dct=inv_dpcm(Y_dct, Cb_dct, Cr_dct,8)
    
    Y_dct, Cb_dct, Cr_dct=inv_quantizacao(Y_dct, Cb_dct, Cr_dct,10)
    
    y_d,cb_d,cr_d=dct_block_decoder(Y_dct, Cb_dct, Cr_dct,8)
    
    global array2
    array2=np.array(y_d)
    
    #7.1.1
    #y_d,cb_d,cr_d=dct_decoder(Y_dct, Cb_dct, Cr_dct)
    
    #6.1
    y,cb,cr=upsampling(y_d,cb_d,cr_d,1)
    
    #5.2
    r,g,b = ycbcr2rgb(y,cb,cr)
    
    #4.1
    r,g,b = inverse_pading(r,g,b, nl, nc)
    
    #3.4
    img=reverse_get_rgb_components(r, g, b)
    
    
    return img


def main():
    plt.close('all')
    
    #3.1
    load_images("bmp")
    
    plt.style.use("dark_background")
    img = imgs[0]
    nl, nc, _ = img.shape

    show_image(img, "Inicial image")
    encoded_y,encoded_cb,encoded_cr = encoder(img)
    
    img_decoded = decoder(encoded_y,encoded_cb,encoded_cr, nl, nc)
    
    #figura 9
    show_image(img_decoded, "Decoded image")
    
    distortion_calc(img,img_decoded)
    getydiff()

if __name__ == "__main__":
    main()
    # Show plots
    plt.show()
