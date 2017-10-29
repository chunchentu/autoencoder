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
	random.shuffle(file_list)

	testNum = int(np.floor(data_size*test_ratio))
	trainNum = data_size - testNum
	print("There are {} files in the source directory".format(data_size))
	print("Test ratio{}, splitting data into {} training, {} testing".format(test_ratio, trainNum, testNum))

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
			source_path = os.path.join(img_directory, img)
			destination_path = os.path.join(train_dir, img)
			os.rename(source_path, destination_path)
		for img in test_file_list:
			source_path = os.path.join(img_directory, img)
			destination_path = os.path.join(test_dir, img)
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



