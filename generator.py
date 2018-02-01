import numpy as np
import scipy.misc as misc
import scipy.ndimage as ndimage
from pycocotools.coco import COCO
from gen_utils import load_image, apply_mask, fit_in_square, prepare_input, prepare_mask

def create_generator(coco, mode, add, object_dim, **kwargs):

	''' Generate training samples.
		coco: MSCOCO object of the associated dataset.
		mode: 'training' or 'testing'.
	 	add:: Specify type of model being used. Ex. ('gist')
		object_dim: Dimension of the object image, channel last.
		context_dim: (Optional) If add == 'gist', the dimension of the contexdt image, channel last.
		testing_size: (Optional) If mode is testing, the number of examples from the testing set to use each epoch.

	'''

	# Make sure that each training and testing set is the same for comparison.
	np.random.seed(7)


	if mode == 'training':

		while True:

			cat_id_index = np.random.choice(range(0,80))

			gt_vector = np.eye(80)[cat_id_index:cat_id_index+1][0]

			image_id = np.random.choice(coco.getImgIds(catIds = coco.getCatIds()[cat_id_index]))

			annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id, catIds=coco.getCatIds()[cat_id_index]))

			object_mask = coco.annToMask(np.random.choice(annotations))


			try:

				object_im = fit_in_square(apply_mask(coco, raw_image = load_image(image_id, mode), mask = object_mask, keep = 'object', crop = 0), object_dim)

				object_input = prepare_input(object_im)

			except: continue

			if add != None:
				
				try: 

					context_dim = kwargs['context_dim']

					context_image = fit_in_square(apply_mask(coco, raw_image = load_image(image_id, mode), mask = object_mask, keep = 'context', crop = 0), kwargs['context_dim'])

					context_input = prepare_input(context_image)

				except: continue

				yield({'object_input': object_input, 'context_input': context_input}, {'gistnet_output': np.array([gt_vector])})

			else:

				yield({'object_input': object_input}, {'objectnet_guess': np.array([gt_vector])})

	if mode == 'testing':

		testing_size = kwargs['testing_size']

		testing_ann_ids = np.random.permutation(coco.getAnnIds())

		if testing_size > len(testing_ann_ids): raise ValueError("Parameter 'testing_size' greater than number of testing samples available.")

		while True:

			count = -1

			while count < testing_size:

				count += 1

				ann_id = int(testing_ann_ids[count])

				ann = coco.loadAnns(ann_id)[0]

				image_id = ann['image_id'] 

				gt_vector = np.eye(80)[coco.getCatIds().index(ann['category_id']):coco.getCatIds().index(ann['category_id'])+1][0]

				object_mask = coco.annToMask(ann)

				try: 

					object_image = fit_in_square(apply_mask(coco, raw_image = load_image(image_id, mode), mask = object_mask, keep = 'object', crop = 0), object_dim)

					object_input = prepare_input(object_image)

				except: continue

				if add != None:

					try: 

						context_image = fit_in_square(apply_mask(coco, load_image(image_id, mode), object_mask, 'context'), kwargs['context_dim'])
						
						context_input = prepare_input(context_image)

					except: continue

					if add == 'gist':

						yield({'object_input': object_input, 'context_input': context_input}, {'gistnet_output': np.array([gt_vector])})

				else:

					yield({'object_input': object_input}, {'objectnet_output': np.array([gt_vector])})

	else: raise ValueError('Must provide a valid generator mode.')




