import cv2
import numpy as np
import os, json
import matplotlib.pyplot as plt


def load_image(path,RGB=True,gray=False):
    gray_flag = 0 if gray else 1
    img = cv2.imread(path, gray_flag)
    if RGB:img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img  



class image_generator_landmarksAwareV2:
    def __init__(self,
                 image,
                 keypoints,
                 rotate_range=0.0,
                 shift_range=0.0,
                 brightnees_range=0.0,
                 Horizontalflip=False,
                 Verticalflip=False,
                 blur_range=1,
                 noise_factor=0.0,
                 zoom_range=0.0,
                 BorderMode=1,
                 filpprobability=3,
                 sharpen=False,
                 noiseprobability=4,
                 target_shape=None):
        
        '''
        image : numpy array of shape (width, height, 3-channel)
        keypoints: list of landmark or keypoints coordinate [x1,y1,x2,y2, ...]


        '''
        self.image = image
        self.img = image.copy() / 255.0
        self.keypoints = keypoints
        self.k=np.array(keypoints)
        self.rotate_range=rotate_range
        self.shift_range = shift_range
        self.brightnees_range=brightnees_range
        self.Horizontalflip=Horizontalflip
        self.Verticalflip=Verticalflip
        self.blur_range=blur_range
        self.sharpen=sharpen
        self.target_shape=target_shape
        self.zoom_range = zoom_range
        self.fillmode=BorderMode
        self.noise_factor=noise_factor
        self.filpprobability=filpprobability
        self.noiseprobability=noiseprobability
        self.h, self.w, self.num_channels = self.image.shape
        self.center = (self.w//2, self.h//2)
        
        # checking for paramters
        if isinstance(rotate_range, (float, int)):
            self.rotate_range = [-rotate_range, rotate_range]
        elif (len(rotate_range) == 2 and
              all(isinstance(val, (float, int)) for val in rotate_range)):
            self.rotate_range = [rotate_range[0], rotate_range[1]]
        else:
            raise ValueError('`rotate_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received: %s' % (rotate_range,))
        #-------------------------------------
        if isinstance(shift_range, (float, int)):
            self.shift_range = [-shift_range, shift_range]
        elif (len(shift_range) == 2 and
              all(isinstance(val, (float, int)) for val in shift_range)):
            self.shift_range = [shift_range[0], shift_range[1]]
        else:
            raise ValueError('`shift_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received: %s' % (shift_range,))
        #--------------------------------------
    
        if isinstance(brightnees_range, (float, int)):
            self.brightnees_range = [1-brightnees_range, 1+brightnees_range]
        elif (len(brightnees_range) == 2 and
              all(isinstance(val, (float, int)) for val in brightnees_range)):
            self.brightnees_range = [brightnees_range[0], brightnees_range[1]]
        else:
            raise ValueError('`brightnees_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received: %s' % (brightnees_range,))   
        #--------------------------------------
        if isinstance(zoom_range, (float, int)):
            self.zoom_range = [1-zoom_range, 1+zoom_range]
        elif (len(zoom_range) == 2 and
              all(isinstance(val, (float, int)) for val in zoom_range)):
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received: %s' % (zoom_range))  
    
    
    def generate(self, numImages=6):
        counter = 0
        while counter != numImages:    
            outputImg, outputKeypts = self._rotate(self.img, self.k)
            #outputImg = self._sharpen(outputImg)
            outputImg, outputKeypts = self._shift(outputImg, outputKeypts)
            outputImg = self._brighten(outputImg)
            if self.Horizontalflip != False:
                if counter % self.filpprobability == 0: 
                    outputImg, outputKeypts = self._flip(outputImg, outputKeypts,'y')
            if self.Verticalflip != False:
                if counter % self.filpprobability+1 == 0: 
                    outputImg, outputKeypts = self._flip(outputImg, outputKeypts,'x')
            outputImg = self._blur(outputImg,)
            if counter % self.noiseprobability==0:outputImg = self._apply_noise(outputImg)
            outputImg, outputKeypts = self._scale_image(outputImg,outputKeypts)
            outputImg, outputKeypts = self._resize(outputImg, outputKeypts)
            outputImg = self.perfect_normlize(outputImg)
            counter += 1
            yield outputImg, outputKeypts
    
    def _rotate(self, img, keypoints):
        angle = np.random.randint(*self.rotate_range)
        radian_angle = (-angle * np.pi) / 180.
        M = cv2.getRotationMatrix2D(self.center, angle, 1)
        rotated_img = cv2.warpAffine(img, M, (self.w,self.h),borderMode=1,flags=cv2.INTER_CUBIC)
        # keypoints augmention
        keypts = keypoints - self.center[0] 
        keypts = np.array([keypts[0::2]*np.cos(radian_angle) - keypts[1::2]*np.sin(radian_angle),
                          keypts[0::2]*np.sin(radian_angle) + keypts[1::2]*np.cos(radian_angle)])
        keypts += self.center[0]
        keypts = np.array([(x,y) for x,y in zip(keypts[0], keypts[1])])
        if np.any(keypts<0):return img, keypoints # or return None
        return rotated_img, keypts.flatten()
    
    def _shift(self, img, keypoints):
        x_shift, y_shift = np.random.uniform(*self.shift_range,size=2)
        x_shift, y_shift = int(self.w * x_shift) ,int(self.h * y_shift)
        M = np.float32([[1,0,x_shift],[0,1,y_shift]])
        shifted_img = cv2.warpAffine(img, M, (self.w,self.h),borderMode=1)
        # keypoints augmentations
        keypts = keypoints
        keypts[::2] = keypts[::2] +  x_shift # the shift in the x-axis
        keypts[1::2] = keypts[1::2] + y_shift
        if np.any(keypts<0):return img, keypoints # or return None
        return shifted_img, keypts
    
    def _brighten(self, img):
        brightness_range = np.random.uniform(*self.brightnees_range)
        return img * brightness_range
    
    
    def _flip(self, img, keypoints, flip_axis):
        keypts= keypoints - self.center[0]        
        if flip_axis=='y':
            M = np.float32([[-1, 0, self.w-1], [0, 1, 0]])
            keypts[::2] = - keypts[::2] 
        else:
            M = np.float32([[1, 0, 0], [0, -1, self.h - 1]])
            keypts[1::2] =  -keypts[1::2] 
            
        hfliped_img = cv2.warpAffine(img, M, (self.w, self.h))
        keypts += self.center[0]
        if np.any(keypts<0):return img, keypoints 
        return hfliped_img, keypts
    
    def _scale_image(self, img, keypoints): 
        ratio = np.random.uniform(*self.zoom_range)
        center_Shift = (1-ratio) * self.center[0]
        M = np.float32([[ratio , 0 , center_Shift],[0, ratio , center_Shift]])
        scaled_img = cv2.warpAffine(img, M, (self.w, self.h),borderMode=1)
        #keypoints augmentaion
        keypts = keypoints
        keypts[0::2] =  keypts[0::2] * ratio + center_Shift
        keypts[1::2] = keypts[1::2] * ratio + center_Shift
        if np.any(keypts<0):return img, keypoints 
        return scaled_img, keypts
    
    def _blur(self,img):
        
        if isinstance(self.blur_range, int):
            k = self.blur_range
        elif len(self.blur_range) >=2 and all(isinstance(val, tuple) for val in self.blur_range):
            k = np.random.choice([val for val,_ in self.blur_range],p=[v for _,v in self.blur_range])
        if k == 0:return img
        kernel=(k, k)
        img = cv2.blur(img, kernel)
        return img
    
    def _apply_noise(self, img):
        if self.noise_factor ==0:return img
        noisy_image = cv2.add(img, self.noise_factor * np.random.randn(*self.image.shape))
        return noisy_image
    def _sharpen(self,img):
        kernel = np.array([[-1, -1, -1], [-1, 9 , -1], [-1, -1, -1]])
        sharpen_img = cv2.filter2D(img, -1, kernel)
        return sharpen_img
    
    def _resize(self, img, keypoints):
        if self.target_shape is None: return img, keypoints
        orignal_shape = self.image.shape
        resized_img = cv2.resize(img, self.target_shape[:2]) 
        keypts = keypoints
        keypts[::2] = keypts[::2] * self.target_shape[1] / float(orignal_shape[1])
        keypts[1::2] =keypts[1::2] * self.target_shape[0] / float(orignal_shape[0])
        return resized_img, keypts
    
    def perfect_normlize(self,image):
        return (image - np.min(image)) / (np.max(image) - np.min(image))
        
        

def display_with_landmark(image,keypoints,color_codes=None):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(1,1,1)
    ax.imshow(image)
    colors = ['r','g','b','y'] if color_codes is None else color_codes
    keypts = [(x,y) for x,y in zip(keypoints[::2],keypoints[1::2])]
    for idx, (point, color) in enumerate(zip(keypts,colors)):
        ax.scatter(*point, c=color)
    plt.show()