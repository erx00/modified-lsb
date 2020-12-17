import numpy as np

from gaborfilter import gaborfilter
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from utils import str_to_binary, binary_to_str
from utils import int_to_binary, binary_to_int


def encode_gabor_lsb(original_img, msg, key, invert=False, sigma=0):
    """
    Encodes a message into the spatial domain specified by a gabor filter. 

    :param original_img: (ndarray) cover image
    :param msg:          (str)     message to encode
    :param key:          (float)   a number [0-360) that will act as the 
                                   oritentation of the gabor filter
    :param invert:       (bool)    if true, encode in non-edge pixels 
    :param sigma:        (int)     apply gaussian blur to image with this sigma value
    :return:             (ndarray) stego image
    """
    if sigma > 20:                                                             #  prevent infinite looping
        return None

    key = (key/180.) * np.pi                                                   #  get orientation in radians
    msg += '<EOS>'
    bits = str_to_binary(msg)
    nbits = len(bits)

    gfilter = gaborfilter(key)              
    img = np.copy(original_img)

    if len(img.shape) == 2:
        img = img.reshape((img.shape[0], img.shape[1], 1))

    nrows, ncols, nchannels = img.shape
    filteredim, thresholds = [], []

    for i in range(nchannels):                                                 #  create edge maps for each channel
        if sigma:   
            blurimg = gaussian_filter(img[:, :, i]/256, sigma)                 #  apply gaussian blur
            filteredim.append(convolve2d(blurimg, gfilter, "same"))
        else: 
            filteredim.append(convolve2d(img[:, :, i]/256, gfilter, "same"))

        thresholds.append(0.75 * np.max(filteredim[i]))                        #  set an arbitrary threshold value for each channel

    bitpos = 0
    for i in range(nrows):
        for j in range(ncols):
            for c in range(nchannels):
                if (not invert and filteredim[c][i, j] < thresholds[c]) \
                    or (invert and filteredim[c][i, j] >= thresholds[c]):      #  skip pixel color if below threshold
                    continue

                if bitpos < nbits:                                             #  only change the LSB of the integer part
                    ipart = int(img[i, j, c])                                  #  to avoid messing with floating point precision
                    fpart = img[i, j, c] - ipart
                    b = int_to_binary(ipart, nbits=64, signed=True)
                    b[-1] = bits[bitpos]
                    img[i, j, c] = binary_to_int(b, signed=True) + fpart
                    bitpos += 1
                else:                                                          #  end of message 
                    return check_match(img, sigma, key, msg, invert, gfilter, filteredim, thresholds)
    
    return check_match(img, sigma, key, msg, invert, gfilter, filteredim, thresholds)  #  end of image
    

def decode_gabor_lsb(original_img, key, invert=False, recurse=True, sigma=0):
    """
    Decodes message from the spatial domain of an image using a gabor filter specified by key.

    :param original_img: (ndarray) stego image
    :param key:          (float)   range: [0, 360) same as the key used in encode
    :param invert:       (bool)    if true, extract bits from non-edge pixels
    :param recurse:      (bool)    recursively call itself with increasing sigmas until a valid
                                   decoding is found
    :param sigma:        (int)     apply gaussian blur with sigma before extracting message
    :return:             (str)     hidden message
    """
    if sigma > 20:                                                             
        return None

    key = (key/180.) * np.pi
    gfilter = gaborfilter(key)
    img = np.copy(original_img)

    if len(img.shape) == 2:
        img = img.reshape((img.shape[0], img.shape[1], 1))
    
    nrows, ncols, nchannels = img.shape
    filteredim, thresholds = [], []

    for c in range(nchannels):
        if sigma:
            blurimg = gaussian_filter(img[:, :, c]/256, sigma)
            filteredim.append(convolve2d(blurimg, gfilter, "same"))
        else: 
            filteredim.append(convolve2d(img[:, :, c]/256, gfilter, "same"))
        
        thresholds.append(0.75 * np.max(filteredim[c]))
    
    pos = 0
    bits, message = [], ""
    
    for i in range(nrows):
        for j in range(ncols):
            for c in range(nchannels):
                if (not invert and filteredim[c][i, j] < thresholds[c]) \
                    or (invert and filteredim[c][i, j] >= thresholds[c]):
                    continue
            
                bits.append(int_to_binary(int(img[i, j, c]), nbits=64, signed=True)[-1])

                if pos % 8 == 7:
                    try:
                        message += binary_to_str(bits)                         #  try to decode the byte
                    except UnicodeDecodeError:
                        if recurse:                                            #  recurse with higher sigma
                            return decode_gabor_lsb(original_img, key, invert, recurse, sigma + 1)
                        return None                                            #  return none to signify 
                                                                               #  invalid decoding
                    bits = []

                pos += 1

                if message[-5:] == '<EOS>':
                    return message[:-5]

    return message[:-5]


def check_match(img, sigma, key, msg, invert, gfilter, filteredim, thresholds):
    """
    Used in encode to ensure that the message in the stego is recoverable by decode.

    :param img:         (ndarray) stego image to be returned
    :param sigma:       (int)     apply gaussian blur with sigma before extracting message
    :param key:         (float)   range: [0, 360) same as the key used in encode
    :param msg:         (str)     message to encode
    :param gfilter:     (ndarray) the gabor filter used in encode to create the stego
    :param filteredim:  (list)    list of convolved channels with the gabor filter
    :param thresholds:  (list)    list of thresholds used for each channel
    :return:            (ndarray) an "invertible" stego 
    """
    decoded_msg = decode_gabor_lsb(img, key, invert, False, sigma)
    if decoded_msg and decoded_msg in msg:
        return img
    return encode_gabor_lsb(img, msg, key, invert, sigma + 1)

    # nchannels = img.shape[2]
    # test_stego = img.copy()/256
    # test_filteredim, test_thresholds = [], []
    # epeaks, dpeaks = [], []

    # for c in range(nchannels):
    #     if sigma:
    #         blurimg = gaussian_filter(test_stego[:, :, c], sigma)
    #         test_filteredim.append(convolve2d(blurimg, gfilter, "same"))
    #     else: 
    #         test_filteredim.append(convolve2d(test_stego[:, :, c], gfilter, "same"))

    #     test_thresholds.append(0.75 * np.max(test_filteredim[c]))
    #     epeaks.append(np.array([i >= thresholds[c] for i in filteredim[c]]).astype("int"))
    #     dpeaks.append(np.array([i >= test_thresholds[c] for i in test_filteredim[c]]).astype("int"))
    
    # if np.all([np.all(epeaks[i] == dpeaks[i]) for i in range(nchannels)]):
    #     print(sigma)
    #     return img
    # return encode_gabor_lsb(img, msg, key, sigma + 1)
