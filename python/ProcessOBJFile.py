import sys
import os

def performOBJtransform(source, target):
	fr = open(source, 'r')
	fw = open(target, 'w')
	while True:
		line = fr.readline()
		if line:
			if line[0:2] == 'v ' or line[0:2] == 'V ' or line[0:2] == 'f ' or line[0:2] == 'F ':
				fw.writelines(line)
		else:
			break
	fr.close()
	fw.close()

if __name__ == '__main__':
	source = 'model_normalized.obj'
	target = 'model_new.obj'
	performOBJtransform(source, target)