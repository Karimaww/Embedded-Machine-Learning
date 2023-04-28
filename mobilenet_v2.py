import argparse
import time

from PIL import Image
from PIL import ImageDraw

import detect
import tflite_runtime.interpreter as tflite
import platform

import os
import numpy as np

EDGETPU_SHARED_LIB = {
	'Linux': 'libedgetpu.so.1',
	'Darwin': 'libedgetpu.1.dylib',
	'Windows': 'edgetpu.dll'
}[platform.system()]


def load_labels(path, encoding='utf-8'):
	"""Loads labels from file (with or without index numbers).

	Args:
	path: path to label file.
	encoding: label file encoding.
	Returns:
	Dictionary mapping indices to labels.
	"""
	with open(path, 'r', encoding=encoding) as f:
		lines = f.readlines()
		if not lines:
			return {}

		if lines[0].split(' ', maxsplit=1)[0].isdigit():
			pairs = [line.split(' ', maxsplit=1) for line in lines]
			return {int(index): label.strip() for index, label in pairs}
		else:
			return {index: line.strip() for index, line in enumerate(lines)}


def make_interpreter(model_file):
	model_file, *device = model_file.split('@')
	return tflite.Interpreter(
		model_path=model_file,
		experimental_delegates=[
			tflite.load_delegate(EDGETPU_SHARED_LIB,
								 {'device': device[0]} if device else {})
		])


def draw_objects(draw, objs, labels):
	"""Draws the bounding box and label for each object."""
	for obj in objs:
		bbox = obj.bbox
		draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
						outline='red')
		draw.text((bbox.xmin + 10, bbox.ymin + 10),
					'%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
					fill='red')

def print_args(args):
	print('-----------------------------------------------------')
	print('Model choisi:\t\t', args.model)
	print('Images:\t\t\t', args.input)
	print('Labeles:\t\t', args.labels)
	print('Seuil:\t\t\t', args.threshold)
	print('Dossier de sauvegarde:\t', args.output)
	print('Nombre d\'inferences:\t', args.count)
	print('-----------------------------------------------------')

def choose_rand_images(images, batchsize):
	return [images[i] for i in np.random.randint(0, len(images), batchsize)]

def main():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-b', '--batch', type=int, default=1,
						help='Batch size for processing inputs.')
	parser.add_argument('-m', '--model', default='models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite',
						help='File path of .tflite file.')
	parser.add_argument('-i', '--input', default='img_test_coco/',
						help='File path of image to process.')
	parser.add_argument('-l', '--labels', default='models/coco_labels.txt',
						help='File path of labels file.')
	parser.add_argument('-t', '--threshold', type=float, default=0.4,
						help='Score threshold for detected objects.')
	parser.add_argument('-o', '--output', default='images_outputs/',
						help='File path for the result image with annotations')
	parser.add_argument('-c', '--count', type=int, default=5,
						help='Number of times to run inference')
	args = parser.parse_args()
	labels = load_labels(args.labels) if args.labels else {}
	interpreter = make_interpreter(args.model)
	interpreter.allocate_tensors()

	print_args(args)
	images = os.listdir(args.input)
	images_list = choose_rand_images(images, args.batch)
	total_time = 0
	for image in images_list:
		print("---Temps d'inference pour l'image " + image + " ---")
		imagename = image
		image = Image.open(args.input + image)
		scale = detect.set_input(interpreter, image.size,
								lambda size: image.resize(size, Image.Resampling.LANCZOS))

		print("Premiere inference est lente car elle inclut le chargement du modele dans la memoire Edge TPU")
		total_time_per_image = 0
		for _ in range(args.count):
			start = time.perf_counter()
			interpreter.invoke()
			inference_time = time.perf_counter() - start
			objs = detect.get_output(interpreter, args.threshold, scale)
			total_time_per_image += inference_time
			print('%.2f ms' % (inference_time * 1000))

		print('La moyenne du temps d\'inference pour l\'image' + imagename + ' est: ' )
		avg_time_per_image = total_time_per_image * 1000 / args.count
		print('%.2f ms' % avg_time_per_image)

		total_time += avg_time_per_image

		if not objs:
			print('No objects detected')

		# if args.output:
		# 	image = image.convert('RGB')
		# 	draw_objects(ImageDraw.Draw(image), objs, labels)
		# 	image.save(args.output + imagename)
		# 	image.show()
	print('-----------------------------------------------------')
	print("Le temps d'inference pour detecter les objets dans " + str(args.batch) + " images est: ")
	print('%.2f ms' % (total_time))
	print('%.2f s' % (total_time / 1000))

if __name__ == '__main__':
	main()
