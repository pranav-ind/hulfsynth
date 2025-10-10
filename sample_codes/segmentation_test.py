from fsl.wrappers.fast import fast as fast

def segment(img_location, output_folder, t = 1, n_classes = 3, g = True, B = True, b=True):
    '''
    img_location : The location to the image that needs to be segmented. (Expecting a HF image)
    output_folder : The folder in which all the outputs of FSL FAST are expected to be stored. (Suggested to use 1 folder per dataset)
    -n,--class	number of tissue-type classes; default=3
    -t,--type	type of image 1=T1, 2=T2, 3=PD; default=T1
    -g,--segments	outputs a separate binary image for each tissue type
    -b		output estimated bias field
	-B		output bias-corrected image
    Reference : For further options : Refer https://fsl.fmrib.ox.ac.uk/fsl/docs/#/structural/fast or Use Command Line : fast 
    '''
    fast(img_location, out= output_folder+'fast', g= g, B=B, n_classes=n_classes, t=t)
    return "Segmentation stored in " + output_folder

path = '/its/home/pi58/projects/hulfsynth/hulfsynth/Data/ixi/T1/102/test/'

img_loc = '/its/home/pi58/projects/hulfsynth/hulfsynth/Data/ixi/T1/102/hf/raw.nii.gz'

segment(img_loc, path)