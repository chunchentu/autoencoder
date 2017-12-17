import os
import random
import numpy as np
# this script generates two disjoint list
# one list shows the training data
# the other shows the testing data


def main(args):
	img_directory = args["img_directory"]
	test_ratio = args["test_ratio"]
	seed = args["seed"]

	file_list = sorted([f for f in os.listdir(img_directory) if not f.startswith('.')])
	data_size = len(file_list)
	random.seed(seed)
	#random.shuffle(file_list)

	testNum = int(np.floor(data_size*test_ratio))
	trainNum = data_size - testNum
	print("There are {} files in the source directory".format(data_size))
	print("Test ratio {}, splitting data into {} training, {} testing".format(test_ratio, trainNum, testNum))

	class_set = set()

	# random select based on class
	for img in file_list:
		img_class = img.split(".")
		class_set.add(int(img_class[0]))

	print("Number of classes:{}".format(len(class_set)))

	train_file_list = []
	test_file_list = []

	for c in class_set:
		class_file_list = [x for x in file_list if x.startswith("{}.".format(c))]
		class_size = len(class_file_list)
		testNum = int(np.floor(class_size*test_ratio))
		trainNum = class_size - testNum
		train_file_list.extend(class_file_list[:trainNum])
		test_file_list.extend(class_file_list[trainNum:])


	train_file_list = sorted(file_list[:trainNum])
	test_file_list = sorted(file_list[trainNum:])

	parent_path = os.path.dirname(img_directory)

	save_file_path = os.path.join(parent_path, "train_file_list.txt")
	with open(save_file_path, "w") as f:
		for l in train_file_list:
			f.write(l+"\n")
	print("The training file list is saved to {}".format(save_file_path))

	save_file_path = os.path.join(parent_path, "test_file_list.txt")
	with open(save_file_path, "w") as f:
		for l in test_file_list:
			f.write(l+"\n")
	print("The testing file list is saved to {}".format(save_file_path))

	train_dir = os.path.join(parent_path, "train_dir")
	if not os.path.isdir(train_dir):
	    print("Folder {} does not exist. The folder is created.".format(train_dir))
	    os.makedirs(train_dir)

	test_dir = os.path.join(parent_path, "test_dir")
	if not os.path.isdir(test_dir):
	    print("Folder {} does not exist. The folder is created.".format(test_dir))
	    os.makedirs(test_dir)

	# ask user if moving data is allowed
	usr_response = input("Do you want to move the images?[y/n]:")

	if usr_response.lower() == "y":
		print("Moving data")
		for img in train_file_list:
			img_class = img.split(".")[0]
			class_dir = os.path.join(train_dir, str(img_class))
			if not os.path.isdir(class_dir):
			    os.makedirs(class_dir)

			source_path = os.path.join(img_directory, img)
			destination_path = os.path.join(class_dir, img)
			os.rename(source_path, destination_path)


		for img in test_file_list:
			img_class = img.split(".")[0]
			class_dir = os.path.join(test_dir, str(img_class))
			if not os.path.isdir(class_dir):
			    os.makedirs(class_dir)

			source_path = os.path.join(img_directory, img)
			destination_path = os.path.join(class_dir, img)
			os.rename(source_path, destination_path)

		print("Done")
	else:
		print("Files are not moved")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_directory", default="", help="The path to the image folder to be separated")
    parser.add_argument("--test_ratio", type=float, default="0.1", help="The ratio of the testing data accounts for")
    parser.add_argument("--seed", type=int, default=1210)
    args = vars(parser.parse_args())

    main(args)



