
import numpy as np
from pycocotools.coco import COCO
from keras.models import Model, load_model
from gen_utils import fit_in_square, apply_mask, prepare_input, prepare_mask, load_image
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import scipy.ndimage as ndimage
import scipy.misc as misc


coco = COCO('scripts/models/peripheralnet_v1/scripts/annotations/instances_val2014.json')

# model = load_model('/n/coxfs01/kevin_wu/pnet_v2/model_weights/add_spotlight_dt_contfrom82_1.h5')
model = load_model('/n/coxfs01/kevin_wu/pnet_v2/model_weights/add_none_025_from100eps_1.h5')
# model = load_model('/n/coxfs01/kevin_wu/peripheralnet/model_weights/objectnet_224_box_blur_small_2_100epoch_valloss.h5')
# model = load_model('/n/coxfs01/kevin_wu/peripheralnet/model_weights/objectnet_224_box_blur_small_2.h5')
# model = load_model('/n/coxfs01/kevin_wu/pnet_v2/model_weights/add_none_1.h5')
# model = load_model('/n/coxfs01/kevin_wu/peripheralnet/model_weights/gistnet_dual_blur_box_448_layer2_1_100ep_valloss.h5.h5')

print(model.summary())

model.compile(
	loss = 'categorical_crossentropy', \
	optimizer = Adam(1e-4, beta_1=.9, beta_2=.99, decay = .3),\
	metrics = ['accuracy']
	)

img_data = np.load('scripts/models/peripheralnet_v1/results/boxes.npy')

object_dim = 224

# add = 'spotlight_dt'
# add = 'gist'
add = None

top1_sum = 0
top3_sum = 0
top5_sum = 0

total_count = 0

all_testing = []

for i in range(np.shape(img_data)[0]):

	print(i)

	img_line = img_data[i]

	image_id = int(img_line[0])
	ann_ids = int(img_line[6])
	annotations = coco.loadAnns(ann_ids)[0]
	gt_vector = coco.getCatIds().index(int(img_line[5]))
	object_mask = coco.annToMask(annotations)

	try:
		obj_im = fit_in_square(apply_mask(coco, raw_image = load_image(image_id), mask = object_mask, keep = 'object', crop = 0.25), object_dim)
	except:
		raise ValueError('wrong')
		continue

	object_input = prepare_input(obj_im)
	# all_testing.append(prepare_input(apply_mask(coco, raw_image = load_image(image_id), mask = object_mask, keep = 'object', crop = 0)))

	if add == None:
		model_pred = model.predict({'object_input': object_input})

	if add == 'gist':
		context_input = prepare_input(fit_in_square(apply_mask(coco, raw_image = load_image(image_id), mask = object_mask, keep = 'context', crop = 0), 448))

		model_pred = model.predict({'object_input': object_input, 'scene_input': context_input})

	if add == 'spotlight_dt':

		context_image = fit_in_square(load_image(image_id), object_dim)
		# misc.imsave('scripts/models/pnet_v2/sample_images/context'+str(image_id)+'.png', np.rollaxis(context_image, 0, 3))
		context_input = prepare_input(context_image)
		object_mask = fit_in_square(prepare_mask(object_mask), object_dim)
		# object_mask_show = object_mask
		# object_mask_show[0] = 255.
		# misc.imsave('scripts/models/pnet_v2/sample_images/mask'+str(image_id)+'.png', np.rollaxis(object_mask_show, 0, 3))
		object_mask = object_mask[0]
		# print(np.shape(np.where(object_mask)))
		# misc.imsave('scripts/models/pnet_v2/sample_images/mask'+str(image_id)+'.png', np.rollaxis(ndimage.distance_transform_cdt(object_mask), 0, 3))
		distance_mask = np.expand_dims(ndimage.distance_transform_cdt(object_mask), axis = 0)
		object_mask = np.expand_dims(object_mask, axis = 0)
		combined_mask = np.expand_dims(np.concatenate((object_mask, distance_mask), axis=0), axis = 0)
		
		model_pred = model.predict({'object_input': object_input, 'spotlight_input': combined_mask, 'context_input': context_input})

	# if add == 'spotlight':

	# 	context_input = prepare_input(fit_in_square(load_image(image_id), object_dim))
	# 	object_mask = fit_in_square(prepare_mask(object_mask), object_dim)[0]
	# 	object_mask = np.expand_dims(np.expand_dims(object_mask, axis = 0), axis = 0)
		
	# 	model_pred = model.predict({'object_input': object_input, 'spotlight_input': object_mask, 'context_input': context_input})

	top5 = model_pred[0].argsort()[-5:]

	top1_sum += int(gt_vector in top5[-1:])
	top3_sum += int(gt_vector in top5[-3:])
	top5_sum += int(gt_vector in top5)
	total_count += 1

	print(total_count)

# np.save('/n/coxfs01/kevin_wu/pnet_v2/model_weights/1000testing.npy', all_testing)

print('25%')
print('Top 1 Average: ' + str(top1_sum*1./total_count))
print('Top 3 Average: ' + str(top3_sum*1./total_count))
print('Top 5 Average: ' + str(top5_sum*1./total_count))








