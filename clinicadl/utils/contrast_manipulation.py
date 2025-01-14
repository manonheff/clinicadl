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

def normalized_value(img : np.array) -> np.array:
	min_value = np.min(img)
	max_value = np.max(img)
	new_img = (img - min_value) / (max_value - min_value)
	return new_img

def smooth_mask(mask : np.array, anomaly_degree: float, sigma: float)-> np.array:
	## from clinicadl generate_utils, based on generate_hypometabolism
	if mask.dtype != float:
		mask = mask.astype(float)
		print("Converted mask to float dtype;")
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
	elif zone == 1 : #right hemisphere
		gm_labels, wm_labels = right_gm_labels, right_wm_labels
	elif zone == 2 : #both hemispheres
		gm_labels = left_gm_labels + right_gm_labels
		wm_labels = left_wm_labels + right_wm_labels
	hemi_gm_mask, hemi_wm_mask = np.isin(seg_img, gm_labels), np.isin(seg_img, wm_labels)

	### randomly choose occipital/frontal zone
	_, max_y, _ = seg_img.shape
	pos_y = round(random.uniform(max_y * (0.5 - fo_margin), max_y * (0.5 + fo_margin)))# fronto-occipital limit within range
	fo_mask = np.zeros_like(seg_img)
	if np.random.random() < 0.5 : #occipital area
		fo_mask[:,0:pos_y,:] = 1
	else : #frontal area
		fo_mask[:,pos_y:,:] = 1
	
	quadran_gm_mask = hemi_gm_mask * fo_mask
	quadran_wm_mask = hemi_wm_mask * fo_mask

	#return (quadran_gm_mask == 1), (quadran_wm_mask == 1)
	return quadran_gm_mask, quadran_wm_mask

def get_full_gm_mask(seg_img : np.array) -> np.array:
	left_gm_labels, right_gm_labels = get_left_right_gm_labels()
	full_gm_mask = np.isin(seg_img, left_gm_labels + right_gm_labels)
	return full_gm_mask


def lower_contrast(brain_filepath : str, seg_filepath : str, anomaly_degree : float, save = True):
	"""
	
	Parameters :
	-------------

	Returns : 
	-------------

	"""

	debug = True

	## load brain
	brain_nifti = nib.load(brain_filepath)
	brain_img = brain_nifti.get_fdata()

	## load segmentation label map
	seg_nifti = nib.load(seg_filepath)
	seg_img = seg_nifti.get_fdata()

	normalized_brain = normalized_value(brain_img) # min max normalization

	full_gm_mask = get_full_gm_mask(seg_img)
	# decrease contrast 
	localised_gm_mask, localised_wm_mask = localised_poor_contrast_mask(seg_img)

	wm = normalized_brain[localised_wm_mask]
	gm = normalized_brain[localised_gm_mask]
	med_percent_diff = 2 * (np.median(gm) - np.median(wm)) / (np.median(gm) + np.median(wm)) * 100
	print("Median percentage difference:",med_percent_diff)

	dilated = dilation(localised_gm_mask, footprint = ball(radius = 2))
	coef = np.random.uniform(1.5, 2.1)
	print("Intensity coef:", coef)
	blurred_localised_gm_mask = smooth_mask(dilated, med_percent_diff * coef, sigma = 2.5)
	blurred_localised_gm_mask *= full_gm_mask # keep only the gm part
	blurred_localised_gm_mask[blurred_localised_gm_mask == 0] = 1 # recreate multiplicative mask

	contrast_mask = gaussian_filter(blurred_localised_gm_mask, sigma=1) 

	if debug : 
		mask_img = nib.Nifti1Image(contrast_mask, brain_nifti.affine, brain_nifti.header)
		nib.save(mask_img, seg_filepath[:-27] + "mask_{}.nii.gz".format(0.5))
		print("Saved final mask")

	low_contrast_img = normalized_brain * contrast_mask
	if save :
		new_brain_img = nib.Nifti1Image(low_contrast_img, brain_nifti.affine, brain_nifti.header)
		nib.save(new_brain_img, seg_filepath[:-27] + "lower_contrast.nii.gz")
		print("Saved new image")

	


if __name__=='__main__':
	print("Entered main function...")
	#brain_filepath = "/network/iss/aramis/datasets/msseg/MSSEG/FLAIR/caps/subjects/sub-MSSEG103/ses-M00/flair_linear/sub-MSSEG103_ses-M00_FLAIR_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_flair.nii.gz"
	#seg_filepath =  "/network/iss/aramis/users/manon.heffernan/synthseg_output/msseg/sub-MSSEG103_ses-M00_FLAIR_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_flair_synthseg.nii.gz"
	
	#brain_filepath = "/network/iss/aramis/datasets/msseg/MSSEG2/caps/subjects/sub-MSSEG2013/ses-M00/flair_linear/sub-MSSEG2013_ses-M00_FLAIR_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_flair.nii.gz"
	#seg_filepath = "/network/iss/aramis/users/manon.heffernan/synthseg_output/msseg2/sub-MSSEG2013_ses-M00_FLAIR_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_flair_synthseg.nii.gz"
	
	brain_filepath = "/network/iss/aramis/datasets/nifd/caps_flair_linear/subjects/sub-NIFD1S0005/ses-M12/flair_linear/sub-NIFD1S0005_ses-M12_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_FLAIR.nii.gz"
	seg_filepath = "/network/iss/aramis/users/manon.heffernan/synthseg_output/nifd/sub-NIFD1S0005_ses-M12_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_FLAIR_synthseg.nii.gz"
	
	anomaly_degree = 50
	lower_contrast(brain_filepath, seg_filepath, int(anomaly_degree), save=True)
	print("Done.")