import torch
import numpy as np 
import matplotlib.pyplot as plt 

# 1. Load the results file, should be a torch tensor 
# of dimension [test_size , 90] for means and 
# [test_size , 90] for std-deviations across all classes
print("loading file..")
# uncert = torch.load('./bayesian_uncertainties_test_true_classes_smaller.pt', map_location=torch.device('cpu'))
drop_means=[]
drop_stds=[]
x_labels = [0.2,0.3,0.4,0.5,0.6,0.7]
for i in x_labels:
	uncert = torch.load(f'./bayesian_uncertainties_test_{i}.pt', map_location=torch.device('cpu'))
	print("done loading.", i)
	test_means = uncert['class_means']
	test_std = uncert['class_std']
	annotations = uncert['true_class']

	# 2. Find all documents with at least 4 predictions above 0.45 to 0.55
	# print(np.where( ((test_means.numpy() > 0.45).sum(1)) > 4 )  )

	# docID = int(input("Which document do you wanna plot?"))
	docID = 1633


	# print(f"Plotting using document {docID}")
	use_log = False
	scaling = 1
	trans = 0
	topK = 10

	print(torch.mean(test_std, dim=0)[1].numpy())
	drop_stds.append(torch.mean(test_std, dim=0)[22].numpy())

	# if use_log:
	# 	means = np.log(test_means[docID, :] * scaling + trans).numpy()
	# 	stds = np.log(test_std[docID, :] * scaling + trans).numpy()
	# else:
	# 	means = (test_means[docID, :] * scaling + trans).numpy()
	# 	stds = (test_std[docID, :] * scaling + trans).numpy()



	# # Select the top k-activations from the 90-vector
	# top_preds = means.argsort()[-topK:][::-1]
	# top_means = means[top_preds]
	# top_stds = stds[top_preds]
	# top_annotations = annotations[docID, :].numpy()[top_preds]
	# drop_means.append(top_means[1])
	# drop_stds.append(top_stds[1])

	# print(top_stds[1])


print(f"Plotting error-plots for document {docID}")

plt.style.use('ggplot')
plt.plot(x_labels, drop_stds)

# f, axarr = plt.subplots(1, sharey = True)

# label = ["blue", "green"]
# GT = ['', '-GT']
# #x_labels = [f"{x}" + GT[annotations[docID, :].tolist()[x]] for x in top_preds]

# ebar = axarr.errorbar(x_labels, 
# 	drop_means, 
# 	drop_stds, 
# 	fmt='ok', 
# 	lw=25,
# 	alpha=0.7,
# 	# ecolor = [label[i] for i in top_annotations], 
# 	marker='.', 
# 	mfc=  "black",
#     mec="black", 
#     ms=4, 
#     mew=3, 
# #    color = "blue",
#     capthick=4
#     )

# axarr.set_title(f"Top {topK} predictions - Document ID: {docID}")
# axarr.set_title(f"Uncertainity vs Dropouts on class 17 - Doc ID: {docID}")
# if(use_log):
# 	plt.axhline(y=np.log(0.5 * scaling), color='r', linestyle='--', label = str(0.5 * scaling) + " in log E")
# 	plt.axhline(y=np.log(0.001 * scaling), color='b', linestyle='--', label = str(0.001 * scaling) + " in log E")
# 	plt.axhline(y=1 * scaling, color='g', linestyle='--', label = str(1 * scaling))

# else:
# 	plt.axhline(y=0.5 * scaling, color='r', linestyle='--', label = str(0.5 * scaling))
# 	plt.axhline(y=0.001 * scaling, color='b', linestyle='--', label = str(0.001 * scaling))
# 	plt.axhline(y=1 * scaling, color='g', linestyle='--', label = str(1 * scaling))

plt.legend()

# if use_log:
# 	axarr.set_ylabel("Predicted mean: log e-space")
# else:
# 	axarr.set_ylabel("Predicted mean")

# # axarr.set_xlabel("Class IDs with corresponding standard deviations")
# axarr.set_xlabel("Dropout rates with corresponding standard deviations")
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

