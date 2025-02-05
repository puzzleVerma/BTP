# Welcome to my BTech Project!

Hi! I'm **Prajjaval**, senior undergrad at IIT Jodhpur!
I have done this project "Detection of defects in ball bearings using Machine Learning" under the supervision of our professor, **Dr B Ravindra** alongwith **Priyanka**.



# Sections

 - Abstarct
 - Introduction
	 - Objective
	 - Ball Bearing
	 - Ball Bearing Dataset
 - Work Done
	 - Methodology
	 - Implementation
 - Results
	 - ANN Model
	 - 1D CNN Model
	 - 1D CNN Multi-kernel Model
	 - 2D CNN Model
 - Conclusion
 - References


# Abstract

Bearings are the most extensively utilised components in rotating machinery, and bearing failures may cause serious malfunctions and even fatalities. According to statistics, bearing failure accounts for over 40% of total induction motor failures and is the major cause of gearbox defects in wind turbines. As a result, diagnosing these bearings becomes critical.

Bearing defect diagnostics has been a challenge in the monitoring of rotating equipment activities, and it has lately received more attention. In order to appropriately categorise defects, fault diagnostic systems often extract information from the waveforms or spectrums of vibration signals. In this assignment, we will apply several deep-learning models to classify bearing defects.

With the rise of automation, Artificial Intelligence (AI) and Deep Learning (DL) are quickly becoming must-have skills for the workforce of the future. Deep Learning educates a computer in a collection of known quantities, and when applied to a set of unknown quantities, the machine (computer) will be able to arrive at a conclusion without any intervention from humans; this is the foundation of Artificial Intelligence.

**In this research, we created four distinct models for detecting and classifying ball bearing faults. We created two datasets containing muted and noisy data points to ensure that the models are robust and can be used in industry. On these datasets, we also test the accuracy of our models.**

The models work effectively in a variety of settings and deliver accurate results.

# Introduction

## Objective

Using various deep learning approaches, we hope to identify faults in ball bearings and categorise the type of defect. To see if the model is industry-ready, we must also see how it performs with noisy and inconsistent data.

## Ball Bearing

A ball bearing consists of an inner ring (IR), an outer ring (OR), a complement of balls, and a separator. The outer diameter of the inner ring (IROD) and the inner diameter of the outer ring (ORID) have a groove in which the balls roll. The groove is commonly called the pathway.

## Ball Bearing Dataset

The dataset which we are going to use is taken from the esteemed **CASE Western Reserve University (Est. 1826)**. This famous dataset is used as a standard in various data-driven machine-learning algorithm papers. The dataset is the result of the project funded by Rockwell Science Center, Office of Naval Research etc.

Data was collected for regular bearings and single-point drive end defects. Data was collected at 48,000 samples/second for drive-end bearing experiments.
Data files are in Matlab format. Each file contains fan and drive end vibration data as well as motor rotational speed.

Link for the dataset:
[CASE Dataset for ball bearing](https://engineering.case.edu/bearingdatacenter/download-data-file)


# Work Done

Preprocessing included:

 - **Concatenation of all fault type files**
	> For future analysis, we will add data from every problem category to a single CSV file. We'll be utilising Python's path feature. This module implements a few practical pathname functions. Open() is used to read or write files, and the os module is used to access the filesystem.

 - **Modified Dataset with muted data points**
	> Here every alternate data point is set to zero to mimic a faulty sensor.

 - **Modified Dataset with noise addition**
	 >Here some noise is added to every data point to test the robustness of our model. This is achieved using the random module of NumPy library. Signal to noise ratio is 0.009

 - **Visualisation of data**
	 >**![](https://lh6.googleusercontent.com/FEdm4aiROZS-304MKZ6q6Fo3mdC_-CPQ_2KiSnuzpjxQSCuoM2Yzu64nkH0cPuukjb4XFq4WEp4U5gBfo-j-nOAZB526AxST6dY3GKLqwIQB13EfsiNBP1OQsaC2fzetYWpD1Lxdkjiy0xRGi2vYJAy2yqtsTVr6OtozicUeaYCOtrmVbvKKx55APIuecg)**
	 **![](https://lh3.googleusercontent.com/VmvOsZ0gtziQrEPNsQUwFPO3Ag_WMMtVEFmmOEQxybig6yGA_ax7laKchxW1O1XghWXjm8rrwID5qmXvmv03t91aiemdsCiv-YVLY_ai3CuGQOiW3Aq6dY7qVV0iLQB2ozfKcEMETB73O2-tT6WqLtfauLKajQ8GFJ4ovJUEPjjXzOMWdTe8wkcsHFO9ug)**

## Theoretical Work

The term "**Artificial Neural Network**" is derived from Biological neural networks that develop the structure of a human brain. Similar to the human brain which has neurons interconnected to one another, artificial neural networks also have neurons that are interconnected to one another in various layers of the networks. These neurons are known as nodes.

An Artificial Neural Network in the field of Artificial intelligence attempts to mimic the network of neurons that makes up a human brain so that computers will have the option to understand things and make decisions in a human-like manner. The artificial neural network is designed by programming computers to behave simply like interconnected brain cells.

**Convolutional Neural Network** is one of the main categories to do image classification and image recognition in neural networks. Scene labelling, objects detections, face recognition, etc., are some of the areas where convolutional neural networks are widely used.

In CNN, each input image will pass through a sequence of convolution layers along with pooling, fully connected layers, and filters (Also known as kernels). After that, we will apply the Soft-max function to classify an object with probabilistic values 0 and 1.


## Implementation

Segmenting the Vibration signal into small portions using a window:

Stride is a parameter of the neural network's filter that modifies the amount of movement over the matrix. We have used a window size of 1000 for our model and a jumping stride of 200.

![](https://lh5.googleusercontent.com/FWgR3Hq7JlQYGYeLCtzydwZa6Xr-uyph6SNe2JHc5iiM3zacQb-yte3ocYPgj0L_ggyWtophUcL_co6hT0VJX4daBoCT2tweDY5XSWO4qYjjkvgOVPm8RyYLiJ3yPhR8qKhydKc7v3NmcO3ESytDq3BdA1FTlghXeXemXdFPgOnqam5x11cJwWm0xEqXbA)

Now we have to convert each instance into a row vector/list and feed this to the neural network to train and the output would be various fault types.

![](https://lh3.googleusercontent.com/FfEhVf-jtwa1QeXWZEdUQCNlJz0wItJjSkW50_Elali7hpWu1LG2MUG3bOBZcKMVNk0cRj7c-yBPLU3cYAgV26mM5yrNwBYD02m0y0huG33J2PUbo4RKW64yhZmCoIFaoWAk1dkud9JBiPdysfAUWcXvncTRkyxul02t84udwP4WMFllueBNdl1vlm0Dtw)

So we will start with performing label encoding on the dataset. Label Encoding refers to converting the labels into a numeric form so as to convert them into a machine-readable form. Machine learning algorithms can then decide in a better way how those labels must be operated.

Then we will split the dataset into train and test sets using the SkLearn library in a 7:3 ratio.

Now we have used Autoencoder to see if raw data is directly separable using t_SNE from the sklearn library. Autoencoders are a specific type of feedforward neural network where the input is the same as the output. They compress the input into a lower-dimensional code and then reconstruct the output from this representation.

![](https://lh3.googleusercontent.com/o1VnvlVHkMDNWFWzN1OlKU28oJMivG4qVTq-zFnqEZKuSn1ZFy4mKEPC34XRjAAOSqyFUjpntqJ0i6KeZijlYzlyZAOh7KrIv1517Glp3wsatv8_iSZ5_y2WmG7MMGoaXuDw9bXz9c7XTxwsVDS4xfKccDgHnP4neZd7KKorKo1YNaBD6ie3ok6m9VTF0Q)

We observe that it is not separable directly from raw data, so we construct our ANN model.

# Result


1.  ANN Model
    

The accuracy and loss for different epoch is given below:

![](https://lh3.googleusercontent.com/p_9sNF8OoXvWwLm9f0z2e3BvFyWA6TOUlFBcIVj-QC6_ctwdNNdLOb1QhLhWjxqVSNAugmKYvET8DIKLAYxm0tyMPeRW_jEcgfLh7tDQ6t51xh0cMOUkD_t1gC23pmPGDf6CaT-kvYdil-FtqyuPKUqKYbBG58LG5YdX8kN0cDPVZj1zkH6H8QzGbCgJ0Q)

The graph for accuracy and val_accuracy is given below:

![](https://lh3.googleusercontent.com/8iZzrEF4kwUj-oOwcTSv8vssPXen38fe9yBRff2Y6MuWnuVJsIEFy90x0xd5fSY2T5PTJzwBbqOj0tbpT7RhfuuOKghTVy8wf2eMd95kc9UmOJRojDJP1CPNTkJ6UFoate7tnYL9z9hCs3MsGJv8SEHmLAlqprKOdG4SvV4U_j6ODXAXZ6zPbUCZJ-5sLQ)

The confusion matrix is given below:

![](https://lh4.googleusercontent.com/ZIDXFvMYH2MV39mKi3AS7jgO_lv6xyObDr1ENw72UcZ3Cm0-NH7U-85b6j8Z_nsLRnJz7m4_hKWkUxLTButVO2wW_bqRKxjto3Xxo2DICW79JWRVVrosEVa9K-4nsZNJ19uZCv-FhzsKYVhB_E1XYePAoNgt84KPePGc9yZkej1uRMezfEPPR_ZlB2MMkw)
<br/><br/>

2.  1D CNN Model
    

The accuracy and loss for different epoch is given below:

![](https://lh3.googleusercontent.com/dws0xkPsd-Uw6Jv95BwX4x5Qxjc_EXzjOKh4QZNSVUU30viC7PA3RAkHA6uwbathpwpIUqDA0Hx1IfRCn-x89NFMMMPSZDv632Td2vwXWZ9ZLmmdmARiskwmFdeycwz7IaIEp9yfD3Sb9zYyBW-o6On3aD96UTMwGjfMXHKDlOmKMeAgdjY_VMWJpuO_Sg)

The graph for accuracy and val_accuracy is given below:

![](https://lh3.googleusercontent.com/PKQw6q3rXNJIibzqpNuxbkMBVprCe0pWbN2ubOMUMp9pIzEZCnz1jAiqqauOx2W_tyd1ohneUHolSoTg4rjgulDvLtfHGGyu-E3mjkalpVrjYnuagMY0QxUTmEP5nLLR7R74UQgcDkSEz4o2jOFNdISkgL82lyjzGW8Py1tZKDvItbYjAd4QurkjZ49qbA)

The confusion matrix is given below:

![](https://lh4.googleusercontent.com/M-t7bU0dUIVfNWTHCA3QvWaPC1wfRZM2-QEU_vz4L875EYoajfw-3vcUmhkLDiVcrESkOL6KOCX1Vq7JLw3LDTLY128cwgpbo4eJ5t6BSjpsItT9SB48zvEz5zK0Gb7Xz9O6qNmVGd--6pyMXHldzenNu4pGNpWZ4b82YFxNaL8UcmrrMfABqYjmslCsWQ)
<br/><br/>
3.  1D CNN Multi-kernel Model
    

The accuracy and loss for different epoch is given below:

![](https://lh6.googleusercontent.com/O3D1556YHGGHQjJ81GE0apd2_2LbyJJW49-9Eojg8QzthwOrRaTXsQ7OQkxKeJhWyyQGfQj0mNmcONRNnu_Qe0KhPLBCaBBb_C9UImSta-ShJZBBGmCE7koNfMr-85spW6M18p6InfpIoGZW6vxKr4I4beSBEV3k-_RgNG8cVFZQOnPCLkvUT38gdJB0qA)

The graph for accuracy and val_accuracy is given below:

![](https://lh6.googleusercontent.com/9qDrMkRtySNp7AsBXtUFQOlDSpJ4IrkDwtYjn465VRHVb4Us3jBQjUSobBnz3Blw1DBQVw7y9E2q4P1hydfEPu-epktiMK6nE0FWbVrMDqytFbUpC80iAHoXQOZL33TWBuGVroWVSXXMxw1Zg65qjR82j3wxMqbX3fNwtT4b3XTkDcNi8GLJ21XHR9iHsA)

The confusion matrix is given below:

![](https://lh5.googleusercontent.com/_reaITaL7J7CHMapn-lZkIdvH5RaZfJdjOm3z4s4tQHwVihdHGYnU75KX9M_GPgkYe6lD9xXY6fLJvPdcep24wZI7UNCxyEau0Gylxittpz4p06udI3up9SOibby0iJFHQsHMHHhYR5DWNbGWlm39wXCaqW4JFojLPM-M2zPrynrLix6CwgTY0wrW0f8tw)
<br/><br/>
4.  2D CNN Model
    

The accuracy and loss for different epoch is given below:

![](https://lh5.googleusercontent.com/ivgVC4Vcwg3BVRUbPIFELk_gzjfu6oLGo43L3KBpSNHTQvifniTTF6aJmyH0G1z6Z-W1i0FsH6I974C2JV1gmbkoTsbkyd2LwXC1aTaU_NgSBmxBBbKnLqT43UBMqkEZXxcAuRS1VH0yyd_IBMAY813TBX-fBmGquLj3bkVmy3PHSlYOwlbDfeYG3W0cQw)

The graph for accuracy and val_accuracy is given below:

![](https://lh6.googleusercontent.com/PinAI--uTgSU32iPuQj61oh8sdXiN21EqL81cUXqVxURrSLs-V4o7aNNLecmmi_lQGWnfg0AarDsJstkb63BKtOB61TK2RQkGjAW9YD-RfKOqcW_BkogD8hZanta42hl2bn8yYDmd21wtyehHg_BDmq0AmUoawIQOLGGV_PQ8WG-H1FJ2zPslA__wvJ37w)

The confusion matrix is given below:

![](https://lh6.googleusercontent.com/JGn-PvdeycgH4MkumX850JLE7jRP6xxk1Fg_w6VTrRWx5Fa-nq6yK5NkvpG76y8hSVn-QsNPOtgBiJ0uJAHqrnI90QQjG-VqNNAdub9ueikfOvhiH1aDvISYfgEcUavu9Oyj1WFj34vNt7dL9yPXFXJzkqumPdMxEaqhGeCigRE0zSp38XO6QjWx-wXcOQ)



# Conclusion


|  Model|Original Dataset|Modified Dataset| Noisy Dataset
|--|--|--|--|
|ANN| 98.98|92.26|82.08|
|1D CNN|99.49|96.16|89.01|
|1D CNN With Multi Kernels|99.74|98.54|82.78
|2D CNN|99.15|94.95|84.26|

# References

 - **Kankar, P.K., Sharma, S.C. and Harsha, S.P., 2011. Fault diagnosis of ball bearings using machine learning methods. Expert Systems with applications, 38(3), pp.1876-1886.**
 - **Bediaga, I., Mendizabal, X., Arnaiz, A. and Munoa, J., 2013. Ball bearing damage detection using traditional signal processing algorithms. IEEE Instrumentation & Measurement Magazine, 16(2), pp.20-25.**
 - [CWRU Code](https://github.com/mohan696matlab/CWRU_Bearing_Fault_Classification)
