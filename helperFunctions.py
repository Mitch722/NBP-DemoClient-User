#  _  _            _               ___ _      _         ___                       _ _   _          
# | \| |_  _ _ __ | |__  ___ _ _  | _ \ |__ _| |_ ___  | _ \___ __ ___  __ _ _ _ (_) |_(_)___ _ _  
# | .` | || | '  \| '_ \/ -_) '_| |  _/ / _` |  _/ -_) |   / -_) _/ _ \/ _` | ' \| |  _| / _ \ ' \ 
# |_|\_|\_,_|_|_|_|_.__/\___|_|   |_| |_\__,_|\__\___| |_|_\___\__\___/\__, |_||_|_|\__|_\___/_||_|
# 

__version__ = '1.14.1' # Major, minor, patch
__author__ = 'Marcin Konowalczyk and Alexander Mitchell'
__email__ = 'aaron@aigaming.com'
__status__ = 'Development'
__doc__ = '''
Helper funcitons for the Number Plates Recognition project
Written by {:s}
Version {:s}
'''.format(__author__,__version__)

import os
import pickle
from copy import deepcopy  # Currently not used
from itertools import product
from scipy import ndimage
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
# from skimage.feature import match_template

FONT_PATH = './UKNumberPlate.ttf'

############################
### GENERAL USEFUL STUFF ###
############################

def setStaticVars(**kwargs):
    ''' Decorator to set the static variables of a function
        https://stackoverflow.com/questions/279561/what-is-the-python-equivalent-of-static-variables-inside-a-function
    '''
    def decorate(func):
        for k in kwargs: setattr(func, k, kwargs[k])
        return func
    return decorate

def findAllInDir(directory, extenstion='.jpg', sort=False, addDir=True):
    ''' Finds all files in the 'directory' with the correct 'extension' '''
    files = list(file for file in os.listdir(directory) if file.endswith(extenstion))
    if sort: files = sorted(files)
    if addDir: files = list(os.path.join(directory, file) for file in files) 
    return files

def edgeDistance(p, size):
    ''' Calculates distance of point 'p' in array of 'size' from the closest edge '''
    dh = min((abs(p[0]), abs(size[0]-p[0])))
    dw = min((abs(p[1]), abs(size[1]-p[1])))
    return dh, dw

def saveToPickle(filename, data):
    ''' Save a single variable to pickle file (*.pkl) '''
    with open(filename, 'wb') as f: pickle.dump(data, f)

def loadFromPickle(filename):
    ''' Load a single variable from pickle file (*.pkl) '''
    with open(filename, 'rb') as f: return pickle.load(f)

class bcolors:
    ''' Print colors
        https://stackoverflow.com/questions/287871/print-in-terminal-with-colors
    '''
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def searchDictioary(dictionary, value):
    ''' Find the key in the dictionary which corresponds to the value '''
    for k in dictionary:
        if dictionary[k] == value: return k

################################
### PIL <-> Numpy conversion ###
################################

def pil2np(im):
    ''' Pillow image to Numpy array '''
    # Return (0,1) numpy array for each channel of im
    if im.mode == 'L':
        return np.asarray(im.getdata(), dtype=np.uint8).reshape((im.size[1], im.size[0]))/255
    elif im.mode == '1':
        return np.asarray(im.getdata(), dtype=np.uint8).reshape((im.size[1], im.size[0]))
    else:
        return list(pil2np(a) for a in im.split())

def np2pil(arr, alpha=None):
    ''' Numpy array to Pillow image '''
    # Check the dimentionality
    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=0)
    elif arr.ndim > 2:
        raise Exception('Invalid dimentionality of the input')
    # Take absolute value (to coersce any processing floating point errors)
    arr = np.abs(arr)
    # Convert to image based on the datatpe
    if arr.dtype == 'bool' or arr.max() <= 1:
        im = Image.fromarray(np.uint8(arr*255))
    elif arr.max() <= 255:
        im = Image.fromarray(np.uint8(arr))
    else:
        raise Exception('Array elements are of invalid size')
    # Add alpha channel by a recursive call to self
    if alpha: im.putalpha(np2pil(alpha, alpha=None))
    return im

##################################
### IMAGE PROCESSING FUNCTIONS ###
##################################

def openGrayScale(imageDir):
    ''' Open grayscale image '''
    return Image.open(imageDir).convert('L')

def gaussBlur(im, radius=2):
    ''' Gaussian blur of the image '''
    if radius > 0: return im.filter(ImageFilter.GaussianBlur(radius=radius))
    return im

def normaliseIntensity(im, cutoff=1):
    ''' Normalise the intensity of the image '''
    return ImageOps.autocontrast(im, cutoff=cutoff)

def binarizeImage(im, threshold=0.5, blur=0):
    ''' Binarise the image and return it as a numpy array '''
    return np2pil(pil2np(gaussBlur(im, blur))>threshold)

def cropContourImg(im, threshold=0.5, blur=0, pad=10):
    ''' Crop the image according to the threshold (with additional pad) '''
    left, upper, right, lower = binarizeImage(im, threshold=threshold, blur=blur).getbbox()
    if pad > 0:
        w, h = im.size
        left = max((0,left-pad))
        right = min((w,right+pad))
        upper = max((0,upper-pad))
        lower = min((h,lower+pad))
    return im.crop((left, upper, right, lower))

def getCorners(im, threshold=0.5, blur=0):
    ''' Get the corners of the numberplate image '''
    im = pil2np(binarizeImage(im, threshold=threshold, blur=blur))
    h, w = im.shape
    f = lambda x,y: x+y
    c = [(h,w),(h,0),(0,0),(0,w)] # Start exactly on the wrong side
    for hi, wi in product(range(h),range(w)):
        if im[hi,wi]:
            if f((h-hi),(w-wi)) > f((h-c[0][0]),(w-c[0][1])): c[0] = (hi,wi)
            if f((h-hi),wi)     > f((h-c[1][0]),   c[1][1] ): c[1] = (hi,wi)
            if f(hi,wi)         > f(   c[2][0] ,   c[2][1] ): c[2] = (hi,wi)
            if f(hi,(w-wi))     > f(   c[3][0] ,(w-c[3][1])): c[3] = (hi,wi)
    return list((p[1],p[0]) for p in c)

def getTransformCoeffs(pa: 'current corners', pb: 'transfrormed corners'):
    ''' Calculate the perspective transfrom coefficients
        https://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil
    '''
    matrix = []
    for p1, p2 in zip(pb, pa):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.array(pa).reshape(8)
    B = np.matrix(matrix, dtype=np.float)

    return np.array(np.dot(np.linalg.inv(B.T * B) * B.T, A)).reshape(8)

def straightenImage(im, c, size=(280,60), pad:int=10, color:int=0):
    ''' Straighten the image, given corners '''
    w, h = size
    c2 = list((x+pad,y+pad) for x,y in [(0, 0), (w,0), (w,h), (0,h)])
    coeffs = getTransformCoeffs(c,c2)
    return im.transform((w+2*pad, h+2*pad), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

def findEdges1(im, blur=0):
    ''' Find edges method 1 '''
    im = gaussBlur(im, blur)
    # Laplacian edge detect
    kernel = (-1, -1, -1,
              -1,  8, -1,
              -1, -1, -1)
    return im.filter(ImageFilter.Kernel((3, 3),kernel,scale=1,offset=255))

def findEdges2(im, blur=0):
    ''' Find edges method 2 '''
    im = gaussBlur(im, blur)
    arr = 2*pil2np(im)-1
    return np2pil(arr)

def downsample(im, factor):
    ''' Downsample by a factor '''
    if factor > 0.0:
        w, h = im.size
        return im.resize((int(w*factor), int(h*factor)),resample=Image.BILINEAR)
    return im

#############################
### GENERATE PLATE IMAGES ###
#############################

@setStaticVars(font=None, size=None)
def getFont(size=100, fontPath=FONT_PATH):
    ''' Open the UK Number Plate Font '''
    if not getFont.font or not getFont.size or getFont.size != size:
        getFont.font = ImageFont.truetype(fontPath, size)
    return getFont.font

def getPlate(plate, height, fontPath=FONT_PATH):
    ''' Gets an image of a 'plate' of a certain height '''
    fontSize = int(1.5*height) # Open font of approx. the right height
    font = getFont(size=fontSize, fontPath=fontPath)
    im = Image.new('L',(len(plate)*fontSize,fontSize),color=255) 
    ImageDraw.Draw(im).text((0,0),plate,font=font,fill=0) # Write up the plate
    arr = pil2np(im)

    # Autocrop the image
    # https://stackoverflow.com/questions/14211340/automatically-cropping-an-image-with-python-pil
    nonEmptyCols = np.where(arr.min(axis=0)<0.5)[0]
    nonEmptyRows = np.where(arr.min(axis=1)<0.5)[0]
    cropBox = [0, arr.shape[0], 0, arr.shape[1]]
    if len(nonEmptyRows) > 0:
        cropBox[0] = min(nonEmptyRows)
        cropBox[1] = max(nonEmptyRows)
    if len(nonEmptyCols) > 0:
        cropBox[2] = min(nonEmptyCols)
        cropBox[3] = max(nonEmptyCols)
    arr = arr[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1]
    im = np2pil(arr)
    size = im.size

    return im.resize((int(size[0]*height/size[1]),height),resample=Image.BILINEAR)

def getPlate2(plate, height, fontSize=None, fontPath=FONT_PATH):
    ''' Gets a numpy array of a 'plate' of a certain height
        Faster than getPlate2
    '''
    # Fudge factor to get the right height of the letters with the default font
    fontSize = fontSize if fontSize else int(1.465*height)
    font = getFont(size=fontSize, fontPath=fontPath)
    im = Image.new('L',(int(len(plate)*height*1.1),int(height*1.1)),color=255)
    ImageDraw.Draw(im).text((0,0),plate,font=font,fill=0) # Write up the plate
    arr = pil2np(im)

    # https://stackoverflow.com/questions/14211340/automatically-cropping-an-image-with-python-pil
    nonEmptyCols = np.where(arr.min(axis=0)<0.5)[0]
    nonEmptyRows = np.where(arr.min(axis=1)<0.5)[0]
    cropBox = [0, arr.shape[0], 0, arr.shape[1]]
    if len(nonEmptyRows) > 0:
        cropBox[0] = min(nonEmptyRows)
        cropBox[1] = max(nonEmptyRows)
    if len(nonEmptyCols) > 0:
        cropBox[2] = min(nonEmptyCols)
        cropBox[3] = max(nonEmptyCols)
    return arr[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1]

################################################
### Root-Mean-Square search though the image ###
################################################

# https://en.wikipedia.org/wiki/Template_matching
# http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html

def RMSSearch(im1, im2, skip=3):
    ''' Find best RMS fit of arrays im2 in im1
        Skips every 'skip' pixels
        im1 must be larger than im2
    '''
    p = [0,0,0] # (score, h, w)
    h, w = im1.shape
    h2, w2 = im2.shape
    for hi, wi in product(range(h-h2)[1::1+skip],range(w-w2)[1::1+skip]):
        v = 1/np.mean((im1[hi:(hi+h2),wi:(wi+w2)]-im2)**2) # 1/MSE error
        if v > p[1]: p = [v, hi, wi]
    return p

def RMSMultiSearch(im1, IM2, names=None, skip=3, keep=None, confFun=None):
    ''' Find best RMS fit of arrays in IM2 to im1 '''
    scores = []
    for i, im2 in enumerate(IM2):
        p = RMSSearch(im1,im2, skip=skip)
        element = (p[0],names[i]) if names else (p[0],i)
        scores.append(element)
    scores.sort(key=lambda x: x[0], reverse=True)
    if keep and keep > 0: scores = scores[:keep] # Keep only some best scores
    f = confFun if confFun else findConfidence1
    confidence = f(scores)
    confidence /= np.sum(confidence)
    return scores, confidence

def normaliseConfidence1(confidence):
    ''' Normaise confidence to be bound between 0 and 1 '''
    return 1-1/(1+abs(confidence))

def findConfidence1(scores, normFun=None):
    ''' Returns a list of confidence values between 0 and 1'''
    # Remove the number plate string from the scores
    scores = list(zip(*scores))[0]
    m = np.mean(scores)
    s = np.std(scores)
    norm = normFun if normFun else normaliseConfidence1
    return list( norm(max((0,score - m))/s) for score in scores )

# def RMSSearch2(im1, im2):
#     ''' Find best RMS fit of arrays im2 in im1
#     Skips every 'skip' pixels
#     '''
#     result = match_template(im1, im2)
#     ij = np.unravel_index(np.argmax(result), result.shape)
#     x, y = ij[::-1]
#     return (np.max(result), y, x)

def findBlurDirection1(im, buffer=20, medfilt=5, theshold=0.5, pad=20):
    ''' Finds the directions of the blur causing the image to be blurred
        im is the CROPPED and STRAIGHTENED image
    '''
    # im = cropContourImg(im, pad=pad)
    # c = getCorners(im)
    # im = straightenImage(im, c)
    imBin = binarizeImage(im, threshold=theshold)
    difference = (pil2np(imBin) - pil2np(im))**2
    if medfilt > 0:
        difference = np2pil(difference)
        difference = difference.filter(ImageFilter.MedianFilter(medfilt))
        difference = pil2np(difference)
    # Select the edges using the 'buffer' distance
    top = difference[0:buffer, buffer:-buffer]
    bottom = difference[-buffer-1:-1, buffer:-buffer]
    left = difference[buffer: -buffer, 0:buffer]
    right = difference[buffer: -buffer, -buffer-1:-1]
    # Concatenate the edges
    longEdge = top + bottom
    shortEdge = left + right
    longEdge = np.sum(longEdge)/np.prod(longEdge.shape)
    shortEdge = np.sum(shortEdge)/np.prod(shortEdge.shape)

    nmb = 0.008 # Fudge factor which works 
    upBlur = longEdge/nmb - 1
    rightBlur = shortEdge/nmb - 1
    return (upBlur, rightBlur)

def makeMotionKernel(size, orientation=True):
    ''' Finds the numpy array that is the kernel for the motion blur '''
    if int(size/2) == size/2: size += 1
    kernel = np.zeros((size, size))
    kernel[int((size - 1) / 2), :] = np.ones(size)
    if orientation == True: kernel = kernel.T / size
    return kernel

def convolveImages(image, kernel):
    ''' Convolves images using ndimage and the correct default inputs '''
    return ndimage.convolve(image, kernel, mode='constant', cval=1.0)

# @setStaticVars(processed={})
# def getLetterAspectRatio(letter, font):
#     ''' Gets the aspect ratio of the letter '''

#     key = letter + str(font.size)
#     if key in getLetterAspectRatio.processed.keys():
#         aspectRatio = getLetterAspectRatio.processed[key]
#     else:
#         im = Image.new('1',(font.size,font.size),color=1)

#         d = ImageDraw.Draw(im)
#         d.text((0,0),letter,font=font,fill=0)
        
#         arr = pil2np(im)

#         # https://stackoverflow.com/questions/14211340/automatically-cropping-an-image-with-python-pil
#         nonEmptyCols = np.where(arr.min(axis=0)==0)[0]
#         nonEmptyRows = np.where(arr.min(axis=1)==0)[0]
#         cropBox = (min(nonEmptyRows), max(nonEmptyRows), min(nonEmptyCols), max(nonEmptyCols))
#         arr = arr[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1]
#         size = np2pil(arr).size


#         aspectRatio = list(x/size[0] for x in size)
#         getLetterAspectRatio.processed[key] = aspectRatio
#     return aspectRatio

# def letterEntropy(im):
#     ''' Calculated the complexity of the letter '''
#     return -np.sum(p*np.log2(p) for p in im.histogram()/np.prod(im.size) if p > 0)

# def letterWidthnes(im, threshold=0.5):
#     ''' Calculates the average width of the letter as a fraction of the width of the image '''
#     return np.mean(np.apply_along_axis( lambda x: np.sum(x<threshold), axis=1, arr=pil2np(im)))/im.width

# def letterVerticalStripiness(im, threshold=(0.5, 0.8)):
#     ''' Calculates how much of the part of the letter is a vertical straight line '''
#     return np.sum(np.apply_along_axis( lambda x: np.sum(x<threshold[0])/im.height, axis=0, arr=pil2np(im))>threshold[1])/im.width

# def letterHorizontalStripiness(im, threshold=(0.5, 0.8)):
#     ''' Calculates how much of the part of the letter is a horizontal straight line '''
#     return np.sum(np.apply_along_axis( lambda x: np.sum(x<threshold[0])/im.width, axis=1, arr=pil2np(im))>threshold[1])/im.height

# def letterStripiness(im, threshold=(0.5, 0.8)):
#     ''' Calculates how much of the part of the letter is a horizontal straight line '''
#     vS = letterVerticalStripiness(im, threshold=threshold)
#     hS = letterHorizontalStripiness(im, threshold=threshold)
#     if hS == 0  and vS == 0: s = np.nan
#     elif vS == 0:            s =  -np.inf
#     elif hS == 0:            s = np.inf
#     elif vS > hS:            s = vS/hS - 1
#     else:                    s = -hS/vS - 1
#     return s

if __name__ == '__main__':
    #This runs when this file is called directly
    for height in range(300)[::5]:
        arr = getPlate2('TEST',height+10)
        print('{}-{}={}'.format(height+10,arr.shape[0],height+10-arr.shape[0]))