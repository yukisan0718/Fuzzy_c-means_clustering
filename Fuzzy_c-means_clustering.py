#!/usr/bin/env python
# coding: utf-8

import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack as fp
from scipy import linalg

### Function for checking empty clusters ###
def check_empty(label, num_clu):
    
    #Get unique label (which must include all number 0~num_clu-1)
    label = np.unique(label)
    
    #Search empty clusters
    emp_ind = []
    for i in range(num_clu):
        if i not in label:
            emp_ind.append(i)
    
    #Output the indices corresponding to the empty clusters
    return emp_ind

### Function for getting k-means clustering ###
def get_KMeans(X, num_clu, max_iter, num_init):
    
    #Define the length of input
    N = int(X.shape[0])  #The number of data
    p = int(X.shape[1])  #The length of feature axis
    
    #For a progress bar
    unit = int(np.floor(num_init/10))
    bar = "#" + " " * int(np.floor(num_init/unit))
    start = time.time()
    
    #Repeat for each trial (initialization)
    minloss = np.inf
    for init in range(num_init):
        
        #Display a progress bar
        print("\rk-means:[{0}] {1}/{2} Processing..".format(bar, init, num_init), end="")
        if init % unit == 0:
            bar = "#" * int(np.ceil(init/unit)) + " " * int(np.floor((num_init-init)/unit))
            print("\rk-means:[{0}] {1}/{2} Processing..".format(bar, init, num_init), end="")
        
        #Initialize label and centroid as random numbers
        label = np.round((num_clu-1) * np.random.rand(N))
        center = np.random.rand(num_clu, p)
        loss = np.inf
        
        #Repeat for each iteration
        for rep in range(max_iter):
            
            #Reset loss value
            new_loss = 0
            
            ### Refinement step (update center) ###
            for j in range(num_clu):
                
                #You can use the following code for concise implementation
                #center[j, :] = X[label==j].mean(axis=0)
                
                #Construct data matrix for the j-th cluster
                clu_X = []
                for i in range(N):
                    #If the i-th data belongs to the j-th cluster
                    if label[i] == j:
                        clu_X.append(X[i, :])
                clu_X = np.array(clu_X)
                
                #Update the j-th centroid
                center[j, :] = np.mean(clu_X, axis=0)
            
            ### Assignment step (update label) ###
            #Initialize valuable for new label vector
            new_label = np.zeros(N)
            
            for i in range(N):
                
                #You can use the following code for concise implementation
                #dist = linalg.norm(X[i, :]-center, axis=1)
                #j = int(np.argsort(dist)[0])
                #new_label[i] = j
                #new_loss = new_loss + dist[j]
                
                #Define the minimum distance
                mindist = np.inf
                
                #Search the closest centroid for the i-th data
                for j in range(num_clu):
                    
                    #Compute the norm (equivalent to sqrt(sum**2))
                    dist = linalg.norm(X[i, :]-center[j, :])
                    
                    #Assign the label corresponding to the minimum distance
                    if dist < mindist:
                        #Update minimum distance
                        mindist = dist
                        new_label[i] = j
                    
                #Get summation of the minimum distances
                new_loss = new_loss + mindist
            
            #Call my function for checking empty clusters
            emp_ind = check_empty(new_label, num_clu)
            if len(emp_ind) > 0:
                for ind in emp_ind:
                    #Assign the same index of data as the one of cluster
                    new_label[ind] = ind
            
            #Get out of the loop if loss and label unchange
            if np.abs(loss-new_loss) < 1e-6 and (new_label == label).all():
                #print("The iteration stopped at {}".format(rep+1))
                break
            
            #Update parameters
            label = np.copy(new_label)
            loss = np.copy(new_loss)
            #print("Loss value: {:.3f}".format(loss))
        
        #Output the result corresponding to minimum loss
        if loss < minloss:
            out_label = np.copy(label)
            out_center = np.copy(center)
            minloss = loss
    
    #Finish the progress bar
    bar = "#" * int(np.ceil(num_init/unit))
    print("\rk-means:[{0}] {1}/{2} {3:.2f}sec Completed!".format(bar, init+1, num_init, time.time()-start), end="")
    print()
    
    #Output the label vector and centroid matrix
    return out_label, out_center, minloss

### Function for getting c-means clustering ###
def get_fuzzyCMeans(X, num_clu, max_iter, num_init, m):
    
    #Fuzzy coefficient m must be more than 1
    if m <= 1:
        m = 1 + 1e-5
    
    #Define the length of input
    N = int(X.shape[0])  #The number of data
    p = int(X.shape[1])  #The length of feature axis
    
    #For a progress bar
    unit = int(np.floor(num_init/10))
    bar = "#" + " " * int(np.floor(num_init/unit))
    start = time.time()
    
    #Repeat for each trial (initialization)
    minloss = np.inf
    for init in range(num_init):
        
        #Display a progress bar
        print("\rc-means:[{0}] {1}/{2} Processing..".format(bar, init, num_init), end="")
        if init % unit == 0:
            bar = "#" * int(np.ceil(init/unit)) + " " * int(np.floor((num_init-init)/unit))
            print("\rc-means:[{0}] {1}/{2} Processing..".format(bar, init, num_init), end="")
        
        #Initialize label and centroid as random numbers
        label = np.random.rand(num_clu, N)
        label = label / np.sum(label, axis=0, keepdims=True)
        center = np.random.rand(num_clu, p)
        loss = np.inf
        
        #Repeat for each iteration
        for rep in range(max_iter):
            
            ### Refinement step (update center) ###
            #Calculate cluster centroids by using weights matrix (label)
            center = ((label**m) @ X) / np.sum(label**m, axis=1, keepdims=True)
            
            ### Assignment step (update label) ###
            #Initialize valuable for fuzzy-label
            new_label = np.zeros((num_clu, N))
            
            #Compute fuzzy-label matrix
            for i in range(N):
                for clu in range(num_clu):
                    #Compute norm for numerator
                    numer_dist = linalg.norm(X[i, :]-center[clu, :])
                    
                    #Get summation of the ratio between norms
                    for j in range(num_clu):
                        #Compute norm for denominator
                        denom_dist = linalg.norm(X[i, :]-center[j, :])
                        if denom_dist < 1e-10:
                            denom_dist = 1e-10
                        new_label[clu, i] = new_label[clu, i] + (numer_dist/denom_dist)**(2/(m-1))
            
            #Avoid zero division
            new_label = np.where(new_label < 1e-10, 1e-10, new_label)
            new_label = new_label**(-1)
            
            #Normalization (it is needed due to error handling)
            new_label = new_label / np.sum(new_label, axis=0, keepdims=True)
            
            #Compute the loss function (generalized mean squares error)
            new_loss = 0
            for i in range(N):
                for j in range(num_clu):
                    #Compute the squared norm as distance
                    dist = linalg.norm(X[i, :]-center[j, :])**2
                    new_loss = new_loss + (new_label[j, i]**m) * dist
            
            #Get out of the loop if loss and label unchange
            if np.abs(loss-new_loss) < 1e-6 and (new_label == label).all():
                #print("The iteration stopped at {}".format(rep+1))
                break
            
            #Update parameters
            label = np.copy(new_label)
            loss = np.copy(new_loss)
            #print("Loss value: {:.3f}".format(loss))
        
        #Output the result corresponding to minimum loss
        if loss < minloss:
            out_label = np.copy(label)
            out_center = np.copy(center)
            minloss = loss
    
    #Finish the progress bar
    bar = "#" * int(np.ceil(num_init/unit))
    print("\rc-means:[{0}] {1}/{2} {3:.2f}sec Completed!".format(bar, init+1, num_init, time.time()-start), end="")
    print()
    
    #Output the label vector and centroid matrix
    return out_label, out_center, minloss

### Main ###
if __name__ == "__main__":
    
    #Setup
    num_clu = 6          #The number of cluster [Default]6
    max_iter = 100       #The number of iteration [Default]100
    num_init = 10        #The number of trial (initialization) [Default]10
    demo = True          #If you use a demo-data (random numbers) [Default]True
    clu_mode = "kmeans"  #Clustering mode (kmeans or cmeans) [Default]kmeans
    m = 2.0              #Using cshape, specify the fuzzy parameter [Default]2.0 (>1.0)
    
    #Define random seed
    np.random.seed(seed=32)
    
    ### Data preparation step ###
    #Generate demo data by using Gaussian
    if demo == True:
        N = 300
        X = np.concatenate([np.random.multivariate_normal([-2, 0], np.eye(2), round(N/num_clu)),
            np.random.multivariate_normal([3, -3], np.eye(2), round(N/num_clu)),
            np.random.multivariate_normal([4, 3], np.eye(2), round(N/num_clu)),
            np.random.multivariate_normal([-1, -4], np.eye(2), round(N/num_clu)),
            np.random.multivariate_normal([-4, -2], np.eye(2), round(N/num_clu)),
            np.random.multivariate_normal([-2, 4], np.eye(2), round(N/num_clu))])
    
    #Using a data file
    elif demo == False:
        
        #Read a data source (csv file)
        with open("./data/covid19_20200425.csv", "r") as f:
            s = f.read().rstrip()
        lines = s.split("\n")
        
        #Define analysis axes
        x_axis, y_axis = 2, 3 #between 2 and 5
        #Analysis_axes = ["Country", "Cases", "Deaths", "Recovered", "Serious", "Tests"]
        Analysis_axes = lines[0].split(",")
        lines = lines[1:]
        print("Horizontal axis: " + str(Analysis_axes[x_axis]) + " / Patients")
        print("Vertical axis: " + str(Analysis_axes[y_axis]) + " / Patients")
        
        #Construct the dataset X and the annotation A
        X, A = [], []
        for i, line in enumerate(lines):
            line = line.split(",")
            #Remove the data if missing
            if line[1]!="" and line[4]!="" and line[x_axis]!="" and line[y_axis]!="":
                x_data = int(line[x_axis]) / int(line[1])
                y_data = int(line[y_axis]) / int(line[1])
                #z_data = int(line[4]) / int(line[1])
                X.append([x_data, y_data])
                A.append(line[0])
        X, A = np.array(X), np.array(A)
    
    #Display graph
    print("Input data shape: {}".format(X.shape))
    plt.rcParams["font.size"] = 16
    plt.figure(figsize=(16, 12))
    plt.scatter(X[:, 0], X[:, 1], c="yellow")
    if demo == True:
        plt.title("Demo-data samples (generated randomly)")
    else:
        #Annotation
        for i, a in enumerate(A):
            plt.annotate(a, xy=(X[i, 0], X[i, 1]), size=10)
        plt.title("COVID-19 data for each country")
        plt.xlabel(Analysis_axes[x_axis] + " / Patients")
        plt.ylabel(Analysis_axes[y_axis] + " / Patients")
    plt.savefig("./result/COVID-19.png")
    
    ### Clustering step ###
    #Normalize input data (indispensable for clustering)
    ave = np.mean(X, axis=0, keepdims=True)
    X = X - ave
    std = np.std(X, axis=0, keepdims=True)
    X = X / std
    
    #Call my function for getting either k-means or fuzzy c-means clustering
    if clu_mode == "kmeans":
        #Call my function for getting k-means clustering
        label, center, loss = get_KMeans(X, num_clu, max_iter, num_init)
    
    elif clu_mode == "cmeans":
        #Call my function for getting fuzzy c-maens clustering
        fuzzy_label, center, loss = get_fuzzyCMeans(X, num_clu, max_iter, num_init, m)
        #print("Fuzzy label: {}".format(fuzzy_label))
        
        #Harden the fuzzy label
        label = np.argmax(fuzzy_label, axis=0)
    
    else:
        print("'clu_mode' must be either 'kmeans' or 'cmeans'.")
        sys.exit()
    
    print("Label: {}".format(label))
    #print("Centroid: {}".format(center))
    print("Loss: {}".format(loss))
    
    #Restore the original scale
    X = (X * std) + ave
    center = (center * std) + ave
    
    #Display graph
    plt.rcParams["font.size"] = 16
    plt.figure(figsize=(16, 12))
    plt.scatter(X[:, 0], X[:, 1], c=label)
    plt.scatter(center[:, 0], center[:, 1], s=250, marker='*', c='red')
    if clu_mode == "kmeans":
        plt.title("A clustering result by k-means")
    elif clu_mode == "cmeans":
        plt.title("A clustering result by fuzzy c-means")
    if demo == False:
        #Annotation
        for i, a in enumerate(A):
            plt.annotate(a, xy=(X[i, 0], X[i, 1]), size=10)
        plt.xlabel(Analysis_axes[x_axis] + " / Patients")
        plt.ylabel(Analysis_axes[y_axis] + " / Patients")
    plt.savefig("./result/clustering.png", dpi=200)