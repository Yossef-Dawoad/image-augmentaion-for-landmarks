# image-augmentation-for-landmarks
python class for image augmentation with **landmarks** or **keypoints** Data in mind

the code design and the functions are similar to waht there in keras API `image_Data_generator` 
most of the functionality and more are coming soon with other improvements

# How to use it 
clone the repo then place the augtool.py file in the project folder

### 1. first initialize the class 
```python
gen = image_generator_landmarksAware(image=img,
                                     keypoints=points,
                                     rotate_range=(-50,70),
                                     shift_range=0.12,
                                     brightness_range=(0.4,1.3),   
                                     noise_range=0.04,
                                     allow_repeate=True,
                                     blur_range=(5,3),
                                     horizontal_flip=True,
                                     zoom_range=(0.8,1.4),
                                     target_shape=(256,256,3),
                                     sharpen=False,
                                     epochs=3
```
the default values for those are 0.0 or False so feel free to leave them empty and just type you want to augment 

### 2.generate 20 augmented images on the Fly
```python 
for image, kpoints in gen.generate(20):
    display_with_landmark(image, kpoints)
```

the ``display_with_landmark`` should be in the augtool so simple `from augtools import *`
or `augtools.display_with_landmark` should be enough.
<hr>
what we just did is generating 20 images and plot them with corresponding landmarks for testing

### 3.save the augmented images to the directory
as you can see we need to save the images to our dirctory with these augmentation
simple python scripting like these can do it for you 
```python
for image, k in gen.generate(20):
    image_name = f'augmented_{uuid.uuid1()}.jpeg' # get random image name
    plt.imsave(f'augmented_images/{image_name}, img) # save the augmented image
    record = {"image_name" : image_name,"coordinates" : lms.tolist()}
    data_list.append(record)

with open('../data/augkeypointsdata.json', 'w') as f:
    json.dump(data_list, f)
```
 
<hr> 

## what will be next
- [ ] improving the code design more
- [ ] more Documentation and examples
- [ ] support for list of images as input instead of one at a time  
- [ ] ability to generate images as batches
- [ ] add more image manipulation functions

### Note: Feel free to report for errors or contribute for improve the code or add new functionality
