import nibabel as nib
import os
import sys
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import binary_dilation as dilation
from skimage.morphology import ball as ball
import random

def get_left_right_wm_labels() -> list:
	"""
	Returns white matter labels on a synthseg segmentation.
	"""
	right_wm = [2, 7]#left hemisphere, left cerebellum synthseg labels
	left_wm = [41, 46] # right hemisphere, right cerebellum synthseg labels
	return left_wm, right_wm

def get_left_right_gm_labels() -> list:
	right_gm =[3, 8, 10, 11, 12, 13, 18, 26] # synthseg labels are inverted compared to the reality, these are synthseg's "right hemisphere" labels
	left_gm = [42, 47, 49, 50, 51, 52, 54, 58]
	return left_gm, right_gm

def get_full_gm_mask(seg_img : np.array) -> np.array:
	left_gm_labels, right_gm_labels = get_left_right_gm_labels()
	full_gm_mask = np.isin(seg_img, left_gm_labels + right_gm_labels)
	return full_gm_mask

def get_full_wm_mask(seg_img : np.array) -> np.array:
	left_wm_labels, right_wm_labels = get_left_right_wm_labels()
	full_wm_mask = np.isin(seg_img, left_wm_labels + right_wm_labels)
	return full_wm_mask

def normalized_value(img : np.array) -> np.array:
	min_value = np.min(img)
	max_value = np.max(img)
	new_img = (img - min_value) / (max_value - min_value)
	return new_img

def smooth_mask(mask : np.array, anomaly_degree: float, sigma: float)-> np.array:
	## from clinicadl generate_utils, based on generate_hypometabolism
	if mask.dtype != float:
		mask = mask.astype(float)
		#print("Converted mask to float dtype;")
	inverse_mask = 1 - 1 * mask
	inverse_mask[inverse_mask == 0] = 1 - anomaly_degree / 100
	gaussian_mask = gaussian_filter(inverse_mask, sigma=sigma)
	return gaussian_mask

def localised_poor_contrast_mask(seg_img: np.array, fo_margin : float = 0.1) -> np.array:
	"""
	
	Parameters :
	--------------
	fo_margin : margin around centerline between occipital and frontal regions. In [0,1].

	Returns :
	--------------
	gm_mask,  : binary mask of the grey matter (full brain)
	localised_gm_mask : binary mask of the grey matter, in chosen zone
	localised_wm_mask : binary mask of the white matter, in chosen zone
	"""
	### randomly choose left/right/both zones
	zone = np.random.randint(0,3)
	left_gm_labels, right_gm_labels = get_left_right_gm_labels()
	left_wm_labels, right_wm_labels = get_left_right_wm_labels()

	gm_labels = []
	wm_labels = []
	if zone == 0 : #left hemisphere
		gm_labels, wm_labels = left_gm_labels, left_wm_labels
		print("left hemisphere")
	elif zone == 1 : #right hemisphere
		gm_labels, wm_labels = right_gm_labels, right_wm_labels
		print('right hemisphere')
	elif zone == 2 : #both hemispheres
		gm_labels = left_gm_labels + right_gm_labels
		wm_labels = left_wm_labels + right_wm_labels
		print("both hemispheres")
	hemi_gm_mask, hemi_wm_mask = np.isin(seg_img, gm_labels), np.isin(seg_img, wm_labels)

	### randomly choose occipital/frontal zone
	_, max_y, _ = seg_img.shape
	pos_y = round(random.uniform(max_y * (0.5 - fo_margin), max_y * (0.5 + fo_margin)))# fronto-occipital limit within range
	fo_mask = np.zeros_like(seg_img)
	if np.random.random() < 0.5 : #occipital area
		fo_mask[:,0:pos_y,:] = 1
		print("occipital area")
	else : #frontal area
		fo_mask[:,pos_y:,:] = 1
		print("frontal area")
	
	quadran_gm_mask = hemi_gm_mask * fo_mask
	quadran_wm_mask = hemi_wm_mask * fo_mask

	return (quadran_gm_mask == 1), (quadran_wm_mask == 1)
	#return quadran_gm_mask, quadran_wm_mask


def lower_contrast(brain_img : np.array, seg_img : np.array, local : bool = False) -> np.array:
	"""
	
	Parameters :
	-------------

	Returns : 
	-------------

	"""
	normalized_brain = normalized_value(brain_img) # min max normalization

	full_gm_mask = get_full_gm_mask(seg_img)
	# decrease contrast 
	if local :
		localised_gm_mask, localised_wm_mask = localised_poor_contrast_mask(seg_img)
	else : 
		localised_gm_mask = full_gm_mask
		localised_wm_mask = get_full_wm_mask(seg_img)

	wm = normalized_brain[localised_wm_mask]
	gm = normalized_brain[localised_gm_mask]
	med_percent_diff = 2 * (np.median(gm) - np.median(wm)) / (np.median(gm) + np.median(wm)) * 100
	print("Median percentage difference:",med_percent_diff)

	dilated = dilation(localised_gm_mask, footprint = ball(radius = 2))
	coef = np.random.uniform(1.5, 2.1)
	#print("Intensity coef:", coef)
	blurred_localised_gm_mask = smooth_mask(dilated, med_percent_diff * coef, sigma = 2.5)
	blurred_localised_gm_mask *= full_gm_mask # keep only the gm part
	blurred_localised_gm_mask[blurred_localised_gm_mask == 0] = 1 # recreate multiplicative mask

	contrast_mask = gaussian_filter(blurred_localised_gm_mask, sigma=1) 

	return contrast_mask