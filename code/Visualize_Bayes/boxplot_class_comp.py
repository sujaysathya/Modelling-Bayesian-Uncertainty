import torch
import numpy as np 
import matplotlib.pyplot as plt 

# 1. Load the results file, should be a torch tensor 
# of dimension [test_size , 90] for means and 
# [test_size , 90] for std-deviations across all classes
print("loading file..")
uncert = torch.load('./bayesian_uncertainties_test_2.pt', map_location=torch.device('cpu'))
print("done loading.")
test_means = uncert['class_means']
test_std = uncert['class_std']

# 2. For each of the rarest and most common classes, 
# gather their predicted means and std-devs

# docID = np.random.choice(range(test_means.size()[0]))
docID = 1446
use_log = True
scaling = 1
trans = 1

common_means = []
common_stds = []
common_indeces = [21, 0, 46, 26, 17] # <-- top 5

if(use_log):
	# common_occurrence = [36.708, 21.418, 7.036, 5.251, 5.183]
	for common_idx in common_indeces:
		common_means.append(np.log(test_means[docID, common_idx] * scaling + trans).tolist())
		common_stds.append(np.log(test_std[docID, common_idx] * scaling + trans).tolist())

	rare_means = []
	rare_stds = []
	# rare_indeces = [58, 5, 42, 28, 52] # <-- bot 5 
	rare_indeces = [65, 77, 67, 55, 74] # <-- "50-55"

	# rare_occurrence = [0.017, 0.017, 0.017, 0.017, 0.000]
	for rare_idx in rare_indeces:
		rare_means.append(np.log(test_means[docID, rare_idx] * scaling + trans).tolist())
		rare_stds.append(np.log(test_std[docID, rare_idx] * scaling + trans).tolist())
else:
	# common_occurrence = [36.708, 21.418, 7.036, 5.251, 5.183]
	for common_idx in common_indeces:
		common_means.append((test_means[docID, common_idx] * scaling + trans).tolist())
		common_stds.append((test_std[docID, common_idx] * scaling + trans).tolist())

	rare_means = []
	rare_stds = []
	# rare_indeces = [58, 5, 42, 28, 52] # <-- bot 5 
	rare_indeces = [65, 77, 67, 55, 74] # <-- "50-55"

	# rare_occurrence = [0.017, 0.017, 0.017, 0.017, 0.000]
	for rare_idx in rare_indeces:
		rare_means.append((test_means[docID, rare_idx] * scaling + trans).tolist())
		rare_stds.append((test_std[docID, rare_idx] * scaling + trans).tolist())



# Hand made plotting for testing
# docID = 1195
# classID = 35
print(f"Plotting error-plots for document {docID}")

plt.style.use('ggplot')
f, axarr = plt.subplots(1, sharey = True)


axarr.errorbar([f"Class: {x}" for x in common_indeces + rare_indeces], 
	common_means + rare_means, 
	common_stds + rare_stds, 
	fmt='ok', 
	lw=4,
	ecolor = ["green"] * 5 + ["red"] * 5, 
	marker='.', mfc=  "blue",
    mec='blue', ms=10, mew=4, 
    color = "blue", 
    # label = common_idx
    )

axarr.set_title(f"5 common/rare classes - Document ID: {docID}")
if(use_log):
	plt.axhline(y=np.log(0.5 * scaling), color='r', linestyle='--', label = str(0.5 * scaling) + " in log E")
	plt.axhline(y=np.log(0.001 * scaling), color='b', linestyle='--', label = str(0.001 * scaling) + " in log E")
else:
	plt.axhline(y=0.5 * scaling, color='r', linestyle='--', label = str(0.5 * scaling))
	plt.axhline(y=0.001 * scaling, color='b', linestyle='--', label = str(0.001 * scaling))

plt.legend()


axarr.set_ylabel("Predicted mean: log e-space")
axarr.set_xlabel("Class IDs with corresponding standard deviations")

plt.show()



# axarr[0].set_ylabel("Predicted mean")

# f.subplots_adjust(hspace=0.5)
# axarr[1].errorbar([f"Class: {x}" for x in rare_indeces], rare_means, rare_stds, 
# 	fmt='ok', 
# 	lw=4,
# 	ecolor = "red", 
# 	marker='.', mfc='red',
#     mec='red', ms=10, mew=4, 
#     color = "blue")

# axarr[1].set_title("5 Most rare classes")


# #	3.2		Plot the uncertainties over the rare classes
# for i in range(5):
# 	y = rare_means[rare_indeces[i]]
# 	std = rare_stds[rare_indeces[i]]
# 	axarr[i + 5].plot(x, y, color = 'red', linewidth = 1)
# 	axarr[i + 5].fill_between(x, y - 2 * std, y + 2 * std, color='red', alpha=0.2)
# 	axarr[i + 5].set_ylim([0, 1])
# 	axarr[i + 5].set_title(f"RARE: Class {rare_indeces[i]} with training occurrence {round(rare_occurrence[i], 2)} %")

