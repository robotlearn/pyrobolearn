## Online Handwritten Datasets

This repo contains the python code to load, parse and plot online handwritten characters.

In this folder, I focus on the following datasets:

1. Character Trajectories Data Set (2,858): https://archive.ics.uci.edu/ml/datasets/Character+Trajectories 
2. UJI Pen Characters (V2) Data Set (11,640): archive.ics.uci.edu/ml/datasets/UJI+Pen+Characters+(Version+2)
3. HWRT Database of Handwritten symbols: www.martin-thoma.de/write-math/data/
4. ICDAR2013 - Handwriting Stroke Recovery from Offline Data (Kaggle): https://www.kaggle.com/c/icdar2013-stroke-recovery-from-offline-data
5. IAM Online Handwriting Database: www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database


There are of course other datasets (mostly offline) that could be used such as:

* MNIST dataset (Offline)
* NIST Handprinted Forms and Characters Database (Offline)
* The IRESTE On/Off Dual Handwriting Database (75€): www.irccyn.ec-nantes.fr/~viardgau/IRONOFF/ICDAR99.htm
* UCI online datasets:
    * Online Handwritten Assamese Characters Dataset (8,235): archive.ics.uci.edu/ml/datasets/Online+Handwritten+Assamese+Characters+Dataset
    * Pen-Based Recognition of Handwritten Digits Dataset (10,992): archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
* CASIA online and offline chinese handwriting Databases: www.nlpr.ia.ac.cn/databases/handwriting/Online_database.html
* CEDAR Handwriting (Offline)


## How to use it?

The whole python code to load and parse the different datasets is inside the jupyter notebook `handwritten-data.ipynb`. To open the jupyter notebook, you will first need to install jupyter. For each dataset, I plotted 25 random samples to give you a general feeling about the dataset. You will need to download the dataset that you which to use (and put it in the corresponding folder if you do not wish to modify the python code).


## FAQs

* What is the meaning of 'online'? Online in this case means that you also have the trajectories (x,y), while offline would mean that you only have pictures of handwritten characters/words/sentences.

* Do I need to cite the dataset that I am using? Yes, you should cite the corresponding paper which can usually be found on the website where you downloaded the dataset.

* For what can I use these datasets? Mostly for demonstration / supervised learning. You could, for instance, use one of the following models to learn the trajectories: Gaussian Processes, Gaussian Mixture Models, Hidden Markov Models, (Deep) Neural Networks, Dynamic Movement Primitives, etc.
                    
