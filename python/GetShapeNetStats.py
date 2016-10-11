import os
import sys
from nltk.corpus import wordnet as wn


def getShapeNetCounts(root, save_name):
	folders = os.listdir(root)
	folders.sort()

	fw = open(save_name, 'w')

	for folder in folders:
		subfolders = os.listdir(root + '/' + folder)
		label = getLable(folder)
		line = folder + ' ' + label + ' ' + str(len(subfolders)) + '\n'
		fw.writelines(line)

	fw.close()


def getCompareCounts(root1, root2, save_name):
	folders = os.listdir(root1)
	folders.sort()

	fw = open(save_name, 'w')

	for folder in folders:
		if not os.path.isdir(root1+'/'+folder) or not os.path.isdir(root2+'/'+folder):
			continue
		subfolders1 = os.listdir(root1 + '/' + folder)
		subfolders2 = os.listdir(root2 + '/' + folder)

		label = getLable(folder)

		portion = float(len(subfolders2)) / len(subfolders1)

		line = '%s %s %d %d %0.2f\n' % (folder, label, len(subfolders1), len(subfolders2), portion)
		#line = folder + ' ' + label + ' ' + str(len(subfolders1)) + ' ' + str(len(subfolders2)) + '\n'
		fw.writelines(line)

	fw.close()

def getLable(id):
	synset = wn._synset_from_pos_and_offset('n', int(id))
	label = str(synset)
	label = label[label.find('\'') + 1: label.find('.')]
	return label

if __name__ == '__main__':
	getShapeNetCounts('/home/yltian/3D/Data/ShapeNetCore.v2', 'classCount.txt')
	getCompareCounts('/home/yltian/3D/Data/ShapeNetCore.v2', \
		'/home/yltian/3D/Data/train', 'compareCount.txt')
