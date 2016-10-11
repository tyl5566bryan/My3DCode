import sys
import os
from nltk.corpus import wordnet as wn

def findCorrespendece(folder, save_name):
	if not folder:
		print("Invalid folder input\n")
		return
	if not save_name:
		print("Invalid save name\n")
		return

	SynsetIds = getSynsetIds(folder)
	Labels = getLabels(SynsetIds)
	outputCorrespendence(SynsetIds, Labels, save_name)

def getSynsetIds(folder):
	SynsetIds = os.listdir(folder)
	SynsetIds.sort()
	return SynsetIds

def getLabels(SynsetIds):
	Labels = []
	for id in SynsetIds:
		synset = wn._synset_from_pos_and_offset('n', int(id))
		label = str(synset)
		label = label[label.find('\'') + 1: label.find('.')]
		Labels.append(label)
	return Labels

def outputCorrespendence(SynsetIds, Labels, save_name):
	fw = open(os.getcwd()+'/'+save_name, 'w')
	for i in xrange(1, len(SynsetIds)):
		line = SynsetIds[i] + ' ' + Labels[i] + '\n'
		fw.writelines(line)
	fw.close()



if __name__ == '__main__':
	findCorrespendece(sys.argv[1], sys.argv[2])