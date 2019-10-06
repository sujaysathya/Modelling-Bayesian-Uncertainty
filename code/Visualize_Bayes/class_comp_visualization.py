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
common_means = {}
common_stds = {}
common_indeces = [21, 0, 46, 26, 17]
# common_occurrence = [36.708, 21.418, 7.036, 5.251, 5.183]

common_thres = 0.5
rare_thres = 0.05
std_dev_thres = 0.01

for common_idx in common_indeces:
	print(f"Common Index {common_idx}")

	common_means[common_idx] = test_means[:, common_idx].numpy()
	thres_idx = np.where(common_means[common_idx] >= common_thres)[0]
	common_stds[common_idx] = test_std[:, common_idx].numpy()
	# print(f"Length before filtering {len(common_means[common_idx])}")
	common_means[common_idx] = common_means[common_idx][thres_idx]
	common_stds[common_idx] = common_stds[common_idx][thres_idx]
	print(f"Documents found above threshold for class {common_idx} = {len(common_means[common_idx])}")
	print(f"std-devs above {std_dev_thres} = {len(np.where(common_stds[common_idx] >= std_dev_thres)[0])}")

	# print(len(common_means[common_idx]))
	# print(len(common_stds[common_idx]))
	print("---------\n")

rare_means = {}
rare_stds = {}
# rare_indeces = [58, 5, 42, 28, 52] # <-- bot 5
rare_indeces = [65, 77, 67, 55, 74]
rare_occurrence = [0.017, 0.017, 0.017, 0.017, 0.000]
for rare_idx in rare_indeces:
	print(f"Rare Index {rare_idx}")
	rare_means[rare_idx] = test_means[:, rare_idx].numpy()
	thres_idx = np.where(rare_means[rare_idx] >= rare_thres)[0]
	rare_stds[rare_idx] = test_std[:, rare_idx].numpy()
	# print(f"Length before filtering {len(rare_means[rare_idx])}")
	rare_means[rare_idx] = rare_means[rare_idx][thres_idx]
	rare_stds[rare_idx] = rare_stds[rare_idx][thres_idx]
	print(f"Documents found above threshold for class {rare_idx} = {len(rare_means[rare_idx])}")
	print(f"std-devs above {std_dev_thres} = {len(np.where(rare_stds[rare_idx] >= std_dev_thres)[0])}")
	
	# print(len(rare_means[rare_idx]))
	# print(len(rare_stds[rare_idx]))
	print("---------\n")



# exit(0)

# 3. Actually do some nice plotting
plt.style.use('ggplot')
 
# x = np.linspace(0, 2 * np.pi, 400)
# y = np.sin(x ** 2)

# x = np.arange(test_means.size()[0])
# f, axarr = plt.subplots(10, sharex=True)
# f, axarr = plt.subplots(5, sharex=True)
f, axarr = plt.subplots(10)

std_dev = 2

# 	3.1 	Plot the uncertainties over the common classes
for i in range(5):
	
	y = common_means[common_indeces[i]]
	x = np.arange(y.size)
	std = common_stds[common_indeces[i]]
	axarr[i].plot(x, y, color = 'green', linewidth = 1)
	axarr[i].fill_between(x, y - std_dev * std, y + std_dev * std, color='green', alpha=0.2)
	# axarr[i].set_ylim([0, 1])
	# axarr[i].set_title(f"COMMON: Class {common_indeces[i]} with training occurrence {round(common_occurrence[i], 2)} %")
	axarr[i].set_title(f"COMMON: Class {common_indeces[i]}")


# # #	3.2		Plot the uncertainties over the rare classes
for i in range(5):
	y = rare_means[rare_indeces[i]]
	x = np.arange(y.size)
	std = rare_stds[rare_indeces[i]]
	axarr[i + 5].plot(x, y, color = 'red', linewidth = 1)
	axarr[i + 5].fill_between(x, y - std_dev * std, y + std_dev * std, color='red', alpha=0.2)
	# axarr[i].plot(x, y, color = 'red', linewidth = 1)
	# axarr[i].fill_between(x, y - std_dev * std, y + std_dev * std, color='red', alpha=0.2)

	axarr[i + 5].set_ylim([0, 1])
	axarr[i + 5].set_title(f"RARE: Class {rare_indeces[i]} with training occurrence {round(rare_occurrence[i], 2)} %")
	axarr[i + 5].set_title(f"RARE: Class {rare_indeces[i]}")
	# axarr[i].set_title(f"RARE: Class {rare_indeces[i]}")


# Set axises 
# plt.xticks(ticks = x)
# plt.xlabel(f"Number of documents that passed threshold prediction-mean: {rare_thres}")
plt.xlabel(f"Number of documents that passed threshold for prediction-mean")
# plt.title('Uncertainty: 5 most common (green) and rare (red)\n',fontsize=5)
f.subplots_adjust(hspace=0.4)
plt.show()
