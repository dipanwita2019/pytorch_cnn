# USAGE
# python train.py --model output/model.pth --plot output/plot.png

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from pyimagesearch.lenet import LeNet
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to output trained model")
ap.add_argument("-p", "--plot", type=str, required=True,
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 10

# define the train and val splits
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT

# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the KMNIST dataset
print("[INFO] loading the KMNIST dataset...")
trainData = KMNIST(root="data", train=True, download=True,
	transform=ToTensor())
testData = KMNIST(root="data", train=False, download=True,
	transform=ToTensor())

# calculate the train/validation split
print("[INFO] generating the train/validation split...")
numTrainSamples = int(len(trainData) * TRAIN_SPLIT)
numValSamples = int(len(trainData) * VAL_SPLIT)
(trainData, valData) = random_split(trainData,
	[numTrainSamples, numValSamples],
	generator=torch.Generator().manual_seed(42))

# initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(trainData, shuffle=True,
	batch_size=BATCH_SIZE)
valDataLoader = DataLoader(valData, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(testData, batch_size=BATCH_SIZE)

# calculate steps per epoch for training and validation set
trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
valSteps = len(valDataLoader.dataset) // BATCH_SIZE

# initialize the LeNet model
print("[INFO] initializing the LeNet model...")
model = LeNet(
	numChannels=1,
	classes=len(trainData.dataset.classes)).to(device)

# initialize our optimizer and loss function
opt = Adam(model.parameters(), lr=INIT_LR)
lossFn = nn.NLLLoss()

# initialize a dictionary to store training history
H = {
	"train_loss": [],
	"train_acc": [],
	"val_loss": [],
	"val_acc": []
}

# measure how long training is going to take
print("[INFO] training the network...")
startTime = time.time()

# loop over our epochs
for e in range(0, EPOCHS):
	# set the model in training mode
	model.train()

	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalValLoss = 0

	# initialize the number of correct predictions in the training
	# and validation step
	trainCorrect = 0
	valCorrect = 0

	# loop over the training set
	for (x, y) in trainDataLoader:
		# send the input to the device
		(x, y) = (x.to(device), y.to(device))

		# perform a forward pass and calculate the training loss
		pred = model(x)
		loss = lossFn(pred, y)

		# zero out the gradients, perform the backpropagation step,
		# and update the weights
		opt.zero_grad()
		loss.backward()
		opt.step()

		# add the loss to the total training loss so far and
		# calculate the number of correct predictions
		totalTrainLoss += loss
		trainCorrect += (pred.argmax(1) == y).type(
			torch.float).sum().item()

	# switch off autograd for evaluation
	with torch.no_grad():
		# set the model in evaluation mode
		model.eval()

		# loop over the validation set
		for (x, y) in valDataLoader:
			# send the input to the device
			(x, y) = (x.to(device), y.to(device))

			# make the predictions and calculate the validation loss
			pred = model(x)
			totalValLoss += lossFn(pred, y)

			# calculate the number of correct predictions
			valCorrect += (pred.argmax(1) == y).type(
				torch.float).sum().item()

	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	avgValLoss = totalValLoss / valSteps

	# calculate the training and validation accuracy
	trainCorrect = trainCorrect / len(trainDataLoader.dataset)
	valCorrect = valCorrect / len(valDataLoader.dataset)

	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["train_acc"].append(trainCorrect)
	H["val_loss"].append(avgValLoss.cpu().detach().numpy())
	H["val_acc"].append(valCorrect)

	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
	print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
		avgTrainLoss, trainCorrect))
	print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
		avgValLoss, valCorrect))

# finish measuring how long training took
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

# we can now evaluate the network on the test set
print("[INFO] evaluating network...")

# turn off autograd for testing evaluation
with torch.no_grad():
	# set the model in evaluation mode
	model.eval()
	
	# initialize a list to store our predictions
	preds = []

	# loop over the test set
	for (x, y) in testDataLoader:
		# send the input to the device
		x = x.to(device)

		# make the predictions and add them to the list
		pred = model(x)
		preds.extend(pred.argmax(axis=1).cpu().numpy())

# generate a classification report
print(classification_report(testData.targets.cpu().numpy(),
	np.array(preds), target_names=testData.classes))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["val_loss"], label="val_loss")
plt.plot(H["train_acc"], label="train_acc")
plt.plot(H["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# serialize the model to disk
torch.save(model, args["model"])

#  data looks like this (x)
"""
tensor([[[[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          ...,
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]]],


        [[[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          ...,
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]]],


        [[[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0078, 0.3373,  ..., 0.0000, 0.0000, 0.0000],
          ...,
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]]],


        ...,


        [[[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          ...,
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]]],


        [[[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          ...,
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]]],


        [[[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          ...,
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
          [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]]]])
          
          
# label looks like this
tensor([7, 2, 9, 2, 3, 7, 8, 7, 9, 4, 8, 7, 3, 5, 5, 3, 3, 8, 6, 5, 2, 1, 6, 9,
        9, 3, 3, 6, 0, 0, 7, 2, 4, 7, 9, 5, 5, 2, 6, 0, 6, 1, 6, 7, 3, 2, 4, 3,
        8, 9, 7, 8, 1, 6, 2, 0, 2, 8, 6, 8, 8, 5, 3, 3])
# prediction looks like this
tensor([[-2.3944, -2.2924, -2.4079, -2.2205, -2.1742, -2.2809, -2.3136, -2.3612,
         -2.2552, -2.3512],
        [-2.3783, -2.3200, -2.3796, -2.2151, -2.1774, -2.2973, -2.3010, -2.3590,
         -2.2749, -2.3440],
        [-2.3612, -2.2959, -2.4190, -2.2250, -2.1814, -2.2830, -2.2964, -2.3734,
         -2.2892, -2.3232],
        [-2.4300, -2.2946, -2.3859, -2.2201, -2.1894, -2.2909, -2.2958, -2.3364,
         -2.2678, -2.3382],
        [-2.3804, -2.3212, -2.3487, -2.2443, -2.1568, -2.3081, -2.3129, -2.3489,
         -2.2618, -2.3637],
        [-2.3953, -2.2939, -2.3961, -2.2069, -2.1481, -2.2843, -2.3087, -2.3912,
         -2.2754, -2.3575],
        [-2.3862, -2.2800, -2.3930, -2.2389, -2.1529, -2.2835, -2.3026, -2.3675,
         -2.2846, -2.3624],
        [-2.4159, -2.2877, -2.4051, -2.2194, -2.1501, -2.2786, -2.2824, -2.3916,
         -2.2817, -2.3460],
        [-2.4139, -2.2822, -2.4204, -2.2337, -2.1456, -2.2764, -2.3411, -2.3566,
         -2.2563, -2.3318],
        [-2.3908, -2.3030, -2.3788, -2.2449, -2.1536, -2.2964, -2.3319, -2.3511,
         -2.2556, -2.3432],
        [-2.3736, -2.3041, -2.3705, -2.2614, -2.1590, -2.2919, -2.3223, -2.3423,
         -2.2723, -2.3473],
        [-2.3855, -2.3469, -2.3730, -2.2452, -2.1567, -2.2873, -2.3000, -2.3571,
         -2.2615, -2.3352],
        [-2.3442, -2.3282, -2.3656, -2.2439, -2.1971, -2.2905, -2.3127, -2.3442,
         -2.2665, -2.3463],
        [-2.3646, -2.3280, -2.3473, -2.2467, -2.1718, -2.2952, -2.3322, -2.3446,
         -2.2698, -2.3422],
        [-2.3678, -2.3266, -2.3571, -2.2413, -2.1833, -2.3091, -2.3320, -2.3303,
         -2.2638, -2.3297],
        [-2.3547, -2.3245, -2.3590, -2.2394, -2.1940, -2.2981, -2.3258, -2.3351,
         -2.2666, -2.3422],
        [-2.3575, -2.2967, -2.3586, -2.2557, -2.1870, -2.2917, -2.2894, -2.3650,
         -2.2668, -2.3737],
        [-2.3918, -2.2962, -2.3652, -2.2481, -2.1795, -2.3001, -2.3134, -2.3467,
         -2.2651, -2.3373],
        [-2.4032, -2.2931, -2.4062, -2.2315, -2.1228, -2.2941, -2.2946, -2.3787,
         -2.2799, -2.3563],
        [-2.3656, -2.2895, -2.3670, -2.2316, -2.2050, -2.2910, -2.3154, -2.3314,
         -2.2913, -2.3516],
        [-2.4036, -2.2820, -2.3969, -2.2268, -2.1889, -2.2643, -2.3129, -2.3603,
         -2.2794, -2.3330],
        [-2.3897, -2.3055, -2.4005, -2.2254, -2.1358, -2.2858, -2.3300, -2.3526,
         -2.2623, -2.3692],
        [-2.4135, -2.2918, -2.4023, -2.2245, -2.1335, -2.2710, -2.3078, -2.3830,
         -2.2649, -2.3691],
        [-2.3668, -2.2958, -2.3733, -2.2426, -2.2119, -2.2676, -2.2936, -2.3667,
         -2.2663, -2.3566],
        [-2.3844, -2.2984, -2.4046, -2.2146, -2.1347, -2.2942, -2.3353, -2.3537,
         -2.2814, -2.3552],
        [-2.3907, -2.2954, -2.3861, -2.2587, -2.1662, -2.2887, -2.3120, -2.3443,
         -2.2647, -2.3393],
        [-2.3739, -2.3028, -2.3522, -2.2520, -2.1849, -2.2995, -2.2956, -2.3602,
         -2.2565, -2.3652],
        [-2.3828, -2.3041, -2.3799, -2.2437, -2.1590, -2.2767, -2.3070, -2.3640,
         -2.2751, -2.3560],
        [-2.3984, -2.3200, -2.3709, -2.2162, -2.1687, -2.3154, -2.3157, -2.3515,
         -2.2744, -2.3167],
        [-2.3967, -2.3130, -2.3729, -2.2105, -2.1579, -2.2953, -2.3483, -2.3497,
         -2.2690, -2.3380],
        [-2.4000, -2.3434, -2.3481, -2.2125, -2.1543, -2.3040, -2.2991, -2.3612,
         -2.2619, -2.3681],
        [-2.4086, -2.2988, -2.4102, -2.2256, -2.1925, -2.2625, -2.2982, -2.3590,
         -2.2705, -2.3235],
        [-2.3940, -2.2895, -2.3854, -2.2459, -2.1804, -2.2834, -2.2938, -2.3616,
         -2.2686, -2.3436],
        [-2.4063, -2.2955, -2.4023, -2.2176, -2.1373, -2.2604, -2.3095, -2.3852,
         -2.2728, -2.3743],
        [-2.4006, -2.3034, -2.3757, -2.2282, -2.1513, -2.2781, -2.2843, -2.3828,
         -2.2918, -2.3570],
        [-2.3602, -2.2915, -2.3617, -2.2564, -2.1819, -2.3049, -2.3103, -2.3497,
         -2.2797, -2.3439],
        [-2.3590, -2.3156, -2.3683, -2.2386, -2.2097, -2.3024, -2.3102, -2.3281,
         -2.2759, -2.3294],
        [-2.3916, -2.3010, -2.3979, -2.2316, -2.1600, -2.2733, -2.2988, -2.3650,
         -2.2888, -2.3427],
        [-2.3666, -2.2770, -2.3910, -2.2458, -2.1903, -2.2746, -2.3214, -2.3592,
         -2.2786, -2.3388],
        [-2.3760, -2.3193, -2.3909, -2.2432, -2.1736, -2.2781, -2.3031, -2.3582,
         -2.2750, -2.3279],
        [-2.3734, -2.3051, -2.3908, -2.2440, -2.1869, -2.2755, -2.3026, -2.3510,
         -2.2654, -2.3494],
        [-2.3523, -2.2873, -2.4005, -2.2617, -2.1741, -2.2776, -2.3144, -2.3620,
         -2.2639, -2.3516],
        [-2.3582, -2.2909, -2.3888, -2.2270, -2.1950, -2.2804, -2.3069, -2.3690,
         -2.2682, -2.3601],
        [-2.4175, -2.3030, -2.4102, -2.2123, -2.1634, -2.2405, -2.3125, -2.3711,
         -2.2774, -2.3500],
        [-2.3889, -2.2847, -2.3496, -2.2359, -2.1826, -2.3023, -2.2989, -2.3623,
         -2.2778, -2.3611],
        [-2.3629, -2.3084, -2.4031, -2.2284, -2.1827, -2.2785, -2.3022, -2.3663,
         -2.2760, -2.3376],
        [-2.4007, -2.3185, -2.3865, -2.2339, -2.1242, -2.2963, -2.3044, -2.3931,
         -2.2654, -2.3356],
        [-2.3699, -2.2915, -2.3971, -2.2422, -2.1801, -2.2683, -2.3292, -2.3373,
         -2.2692, -2.3611],
        [-2.3764, -2.2830, -2.4141, -2.2241, -2.1629, -2.2813, -2.3066, -2.3636,
         -2.3024, -2.3363],
        [-2.3699, -2.3232, -2.4126, -2.2132, -2.1599, -2.2836, -2.2840, -2.3805,
         -2.2908, -2.3352],
        [-2.3922, -2.2817, -2.4150, -2.2311, -2.1654, -2.2699, -2.2987, -2.3623,
         -2.2905, -2.3449],
        [-2.3852, -2.2827, -2.3711, -2.2143, -2.1952, -2.3072, -2.3090, -2.3373,
         -2.2958, -2.3453],
        [-2.3809, -2.3133, -2.4016, -2.2201, -2.1333, -2.2836, -2.3279, -2.3825,
         -2.2640, -2.3506],
        [-2.3654, -2.3003, -2.3967, -2.2289, -2.1775, -2.2665, -2.3200, -2.3654,
         -2.2623, -2.3653],
        [-2.3642, -2.2970, -2.4084, -2.2210, -2.1803, -2.2780, -2.3015, -2.3723,
         -2.2765, -2.3490],
        [-2.3848, -2.3134, -2.3682, -2.2197, -2.1999, -2.3072, -2.3261, -2.3324,
         -2.2856, -2.3037],
        [-2.3446, -2.2956, -2.3445, -2.2679, -2.2093, -2.2895, -2.3067, -2.3468,
         -2.2732, -2.3577],
        [-2.3844, -2.2766, -2.4241, -2.2379, -2.1736, -2.2557, -2.2836, -2.3653,
         -2.2867, -2.3647],
        [-2.3747, -2.2991, -2.3615, -2.2700, -2.1420, -2.2708, -2.3161, -2.3657,
         -2.2671, -2.3836],
        [-2.3642, -2.3007, -2.3916, -2.2221, -2.2049, -2.2799, -2.3201, -2.3496,
         -2.2790, -2.3299],
        [-2.3941, -2.2899, -2.3911, -2.2341, -2.1627, -2.2691, -2.3206, -2.3594,
         -2.2745, -2.3550],
        [-2.3519, -2.3321, -2.3624, -2.2422, -2.1942, -2.2935, -2.3253, -2.3337,
         -2.2715, -2.3321],
        [-2.3713, -2.2842, -2.3529, -2.2567, -2.1825, -2.3128, -2.3114, -2.3310,
         -2.2773, -2.3606],
        [-2.3476, -2.3130, -2.3367, -2.2630, -2.1870, -2.2998, -2.3037, -2.3494,
         -2.2723, -2.3663]], grad_fn=<LogSoftmaxBackward>)
# loss looks like this 
tensor(2.3142, grad_fn=<NllLossBackward>)
"""