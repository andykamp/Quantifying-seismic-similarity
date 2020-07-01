''' This module contains code to handle data '''
import random
import os
import numpy as np
import scipy.ndimage
from PIL import Image
import scipy
import sys
import matplotlib

import matplotlib.pyplot as plt
from skimage.util import view_as_blocks
from sklearn.feature_extraction import image
import tensorflow as tf

import sys, os
sys.path.insert(0, os.path.abspath('..'))
# declare parent dir name 
dirname = sys.path[0] # parent directory
from scripts.data_utils.grid_utils import blockshaped, unblockshaped, plot_embeddings
from scripts.data_utils.augmentation import augment


class SeismicHandler(object):

    ''' Provides a convenient interface to manipulate F3 data '''

    def __init__(self):
        self.datasets = {
            "F3": {
                "path": os.path.join(dirname,"data/processed/f3_entire_int8.npy"),
                "dims": {"inline": 651, "xline": 951, "tline":462}
            }
        }
        # Download data if needed
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.load_dataset()

    def load_dataset(self):
        def params_to_filename(params):
            """ Takes in parameters and retireves the file location """
            filename = self.datasets[params["dataset"]]["path"].split(".")[0]
            filename = f'{filename}_{params["direction"]}.npy'
            return filename


        # generates training data in all, one, or multiple directions
        # options for cropping size, num_crops_per_slice and  augmentation can be set 
        def generate_training_data(output_filename, dataset, direction="iline"):
            # get data to use 
            seismic = np.load(self.datasets[dataset]["path"])#"/Users/anderskampenes/Documents/Dokumenter/NTNU/MASTER/code/data/raw/f3-benchmark/test_once/test1_seismic.npy")
            directions, inline_samples, xline_samples, tline_samples,data, labels, num_samples, shape = generate_directional_dataset(seismic)
            # return the relevant direction 
            return  data[direction],labels[direction]


        def split_data(data, labels, val_split=0.1, test_split=0.1):
            """ convert loaded data into training, validation, and test set, and provide indexes as labels """

            #data = np.random.shuffle(data)
            
            X_train_tot = data[:int(data.shape[0]*(1-test_split))]
            y_train_tot = labels[:int(labels.shape[0]*(1-test_split))]
            X_test = data[ int(data.shape[0]*(1-test_split)):]
            y_test = labels[int(labels.shape[0]*(1-test_split)):]
             # We reserve the last 10000 training examples for validation.
            X_train, X_val = X_train_tot[ :int(X_train_tot.shape[0]*(1-val_split))], X_train_tot[ int(X_train_tot.shape[0]*(1-val_split)):]
            y_train, y_val = y_train_tot[:int(y_train_tot.shape[0]*(1-val_split))], y_train_tot[int(y_train_tot.shape[0]*(1-val_split)):]
            print("Sequence of indexes used as labels for train, val and test: ",  y_train[-1], y_val[-1], y_test[-1])
            
            def plot_training_split():
                """ Plot the training split TODO ON ILINE/TLINE/XLINE """
                fig, ax = plt.subplots()
                plt.xlim(0, 1)
                plt.ylim(0, 1)

                # define splits
                split_1 = (1-test_split)*(1-val_split)
                split_2 = (1-test_split)*(1-val_split) + (1-test_split)* val_split

                # plot colors for split
                plt.axvspan(0, split_1, facecolor='r', alpha=0.5)
                plt.axvspan(split_1, split_2, facecolor='g', alpha=0.5)
                plt.axvspan(split_2, 1, facecolor='b', alpha=0.5)

                # set labels (need dummy data to match legend wth color so plotting something outside of the view)
                x=[0,0]
                y=[0,0]
                _, = plt.plot([10,10,10], label='Trian', color="r")
                __, = plt.plot([10,10,10], label='Val',color="g")
                ___, = plt.plot([10,10,10], label='Test',color="b")

                plt.legend([_,__,___],["Trian", "Val", "Test"])


                plt.show()

            return X_train, y_train, X_val, y_val, X_test, y_test


        # We then define functions for loading seimisc images and infer labels (indexes).
        def get_seismic_data(params):
            filename = params_to_filename(params)
            # if there is not processed data at the filepath, we generate it from the seismic cube
            if not os.path.exists(filename):
                print(f'No data found at {filename}. Generating new from seimsic cube')
                return generate_training_data(filename, params["dataset"],params["direction"])
            print(f'Found data at {filename}. Starting to load it..')
            data = np.load(filename)
            print("done loading data")
            return data


        # We can now download and read the training and test set images and labels.
        filename = os.path.join(dirname,"data/processed/f3-inlines.npy")
        # we can request to retrieve a given dataset given parameters.
        # If we have not already created a dataset matchinf the parameters, we create and store one 
        params={
            "dataset": "F3", 
            "direction": "inline", # oneOf ["iline", "xline","tline", "full"]
        }
        data, labels = get_seismic_data(params)
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(data, labels)
        # We just return all the arrays in order, as expected in main().
        # (It doesn't matter how we do this as long as we can read them again.)
        return X_train, y_train, X_val, y_val, X_test, y_test
       
    

    def process_batch(self, batch, labels, batch_size, image_size, patch_size,  stride = 32):
        """ Take in a batch of cropped images and devides itinto patches """
        # convert the greyscale to rgb by repeating the greyscal 3 times 
        batch = np.repeat(batch[..., np.newaxis], 3, -1)

        # generate 64*64 patches with 32 overlap
        # we do this by moving a lisding window 32 units and cropping       
        if(int(patch_size/stride) ==1): n = int(image_size/stride)# -1)
        else: n = int(image_size/stride) -1
        nn = n*n
        patches = []
        y = []
        idx = 0
        for img in range(batch_size):
            for i in range(n):
                for j in range(n):
                    patches.append(batch[ img, i*stride:i*stride+patch_size, j*stride:j*stride+patch_size,:])
                    y.append(labels[idx])
            idx+=1
                
        batch = np.asarray(patches)
        labels = np.asarray(y)
        return batch, labels


    def get_batch(self, subset, batch_size, image_size=256, patch_size=64,  stride = 32, num_crops=2, augmentation=False ):
        """ Retrieve a batch of images and process each one """ 
        # Select a subset
        if subset == 'train':
            X = self.X_train
            y = self.y_train
        elif subset == 'valid':
            X = self.X_val
            y = self.y_val
        elif subset == 'test':
            X = self.X_test
            y = self.y_test

        # Random choice of samples
        idx = np.random.choice(X.shape[0], batch_size)
        batch = X[idx, :, :]
        labels = y[idx]
        


        # we first crop the image 
        batch, labels = self.crop_image_batch(batch, labels, batch_size, num_crops, image_size )

        batch_size = batch_size*num_crops
        if(augmentation): batch = self.aug_image_batch(batch)
        # we now have our selected batch and want to to create patches  on the fly for each batch_size of them
        # Process batch
        batch, labels = self.process_batch(batch, labels, batch_size, image_size, patch_size,  stride)
        # TODO LABELS SHOULD BE BASED ON INDEX AND/OR WHAT PART OF SLICE!
        return batch.astype('float32'), labels.astype('int32')

    def get_image_batch(self, subset, batch_size, image_size=256,):

        # Select a subset
        if subset == 'train':
            X = self.X_train
            y = self.y_train
        elif subset == 'valid':
            X = self.X_val
            y = self.y_val
        elif subset == 'test':
            X = self.X_test
            y = self.y_test
        
        # extraxt random image 
        idx = np.random.choice(X.shape[0], batch_size)
        batch = X[idx, :, :]
        labels = y[idx]
        return batch.astype('float32'), labels.astype('int32')
    
    
    def crop_image_batch(self, batch, labels, batch_size, num_crops, crop_size):
            cropped_batch = np.empty([batch_size*num_crops, crop_size, crop_size])
            cropped_labels = np.empty([batch_size*num_crops,1])

            for i in range(batch_size):
                 cropped_batch[i*num_crops:i*num_crops+num_crops] = crop_image( batch[i],  num_crops, crop_size)
                 cropped_labels[i*num_crops:i*num_crops+num_crops] = labels[i]
            return cropped_batch.astype('float32'), cropped_labels.astype('int32')

    def aug_image_batch(self, batch):
            for i in range(batch.shape[0]):
                 batch[i] = augment(batch[i])
            return batch.astype('float32')


    def get_n_samples(self, subset):

        if subset == 'train':
            y_len = self.y_train.shape[0]
        elif subset == 'valid':
            y_len = self.y_val.shape[0]
        elif subset == 'test':
            y_len = self.y_test.shape[0]

        return y_len

class SeismicImageGenerator(object):

    ''' Data generator providing a batch of cropped images with augmentation  '''

    def __init__(self, batch_size, subset, image_size=256, num_crops=2, augmentation=False):

        # Set params
        self.batch_size = batch_size
        self.subset = subset
        self.image_size = image_size
        self.augmentation = augmentation
        self.num_crops = num_crops
        # Initialize MNIST dataset
        self.seismic_handler = SeismicHandler()
        self.n_samples = self.seismic_handler.get_n_samples(subset)
        self.n_batches = self.n_samples // batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):

        # Get image
        x, y = self.seismic_handler.get_image_batch(self.subset, self.batch_size, self.image_size)
        x, y = self.seismic_handler.crop_image_batch(x, y, self.batch_size, self.num_crops, self.image_size )
        if(self.augmentation): x = self.seismic_handler.aug_image_batch(x)
        return x, y

class SeismicGenerator(object):

    ''' Data generator that crops seimsic slices to a fixed size, augments each crop, patches it a grid, and devides the columns in the grid into label/pred'''

    def __init__(self, batch_size, subset, image_size=256, patch_size=64, stride=32,  num_crops=2, augmentation=False, sequence=False,  positive_samples=1, terms=4, predict_terms=4, verbose=False):

        # Set params
        self.batch_size = batch_size
        self.subset = subset
        self.image_size = image_size
        self.patch_size= patch_size
        self.stride=stride
        self.augmentation = augmentation
        self.num_crops = num_crops
        self.verbose = verbose

        # for the sequence 
        self.sequence = sequence
        self.positive_samples = positive_samples
        self.terms = terms
        self.predict_terms = predict_terms

        # Initialize SEIMSIC dataset handler
        self.seismic_handler = SeismicHandler()
        self.n_samples = self.seismic_handler.get_n_samples(subset)
        self.n_batches = self.n_samples // batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):

        # Get data
        x, y = self.seismic_handler.get_batch(self.subset, self.batch_size, self.image_size, self.patch_size, self.stride,  self.num_crops, self.augmentation )
        # devide the patched images into sequences if specified 
        if(self.sequence):
            batch_size = self.batch_size*self.num_crops
            X, Y, Y_label = self.create_sequence( x, batch_size, verbose=self.verbose)
            if(self.verbose):print("Created sequence: ", X.shape, Y.shape, Y_label.shape, Y_label)
            return [X, Y], Y_label
        else:
            return x, y
    
    def create_sequence(self, x, batch_size, verbose=False):
         # if stride is not the same as patch dim, then we assume it is halve of dim and have to subtract 1 from number of patches 
        if(int(self.patch_size/self.stride) ==1): n = int(self.image_size/self.stride)# -1)
        else: n = int(self.image_size/self.stride) -1
        nn = n*n
        #print("n", n, nn)

        # extract the rows to predict z+x
        # allocate a array with each colum [column number, rows, patch_size, patch_size, channels]
        sequence = np.empty([batch_size*n, n, x.shape[1], x.shape[2], x.shape[3]]) # TODO DYNAMIC CCOLUMN SIZE NOT 7
        # loop though each batch 
        counter = 0
        for i in range(batch_size):
            # print full image
            if(verbose):plot_embeddings(x[i*nn:i*nn+nn].reshape(nn, x.shape[1]*x.shape[2],x.shape[3]), self.patch_size, self.stride, image_size=self.image_size, channels=3)

            # and for each batch we extract every column
            for j in range(n):
                
                # print extracted column
                sequence[counter]= x[i*nn+j:i*nn+nn +j:n]
                counter +=1

      
        # generate negative predictive terms indexes 
        K = self.predict_terms
        # for random row
        nl = []
        nrr = []
        nrri = []
        for i in range(K):
            nlist = np.arange(0, n)
            nlist = nlist[nlist != (n-K+i)]
            nl.append(nlist)
            nrr.append([sorted(nl[i], key=lambda k: random.random()) for j in range(batch_size*n)])
        nrri = [np.stack([nrr[j][i][0] for j in range(K)], axis=0) for i in range(batch_size*n)]
        
        

        Y = [] # the actual predictive terms 
        Y_label = np.zeros((batch_size*n), dtype=np.float32) # the label of a predictive term, and hence for the entire column/sequence
        n_p = batch_size*n // 2
        for i in range(batch_size*n):
            if i <= n_p:
                Y.append(sequence[i, -K:, ...])
                Y_label[i] = 1
            else:
                 Y.append(sequence[i, nrri[i]])
        Y = np.asarray(Y)
        
        # verify by stacking columns to form a image 
        def stich_rows(n, terms, preds):
            fig = plt.figure(constrained_layout=True)
            widths = [1]*n
            heights = [1]*n
            spec5 = fig.add_gridspec(ncols=n, nrows=n, width_ratios=widths,
                                    height_ratios=heights)
            for row in range(n):
                for col in range(n):
                    ax = fig.add_subplot(spec5[row, col])
                    
                    #plot overlay for the terms to display label 
                    if row>=n-self.predict_terms:
                        plt.imshow(preds[col][row-(n-self.predict_terms)].astype('uint8'))
                        plt.imshow(np.ones(sequence[col][-self.predict_terms].shape)*255, alpha=0.5)
                    else:
                        plt.imshow(terms[col][row].astype('uint8'))

                    #label = 'Width: {}\nHeight: {}'.format(widths[col], heights[row])
                    #ax.annotate(label, (0.1, 0.5), xycoords='axes fraction', va='center')
                    plt.axis('off')
            plt.show()


        # Assemble batch
        X = sequence[:, :-self.predict_terms, ...]
     
        if(verbose):
            for i in range(batch_size//2-2, batch_size//2+2):
                stich_rows(n , X[i*n:i*n+n], Y[i*n:i*n+n])
                # plot a random negative samples
        return X, Y, Y_label


class SequencePathGenerator(object):

    ''' Data generator providing a sequence of rows to predict on. Splits 50/50 positives and negatives '''

    def __init__(self, batch_size, subset, terms, positive_samples=1, predict_terms=1, image_size=256, patch_size=64, stride=32, color=False, rescale=True):

        # Set params
        self.positive_samples = positive_samples
        self.predict_terms = predict_terms
        self.batch_size = batch_size
        self.subset = subset
        self.terms = terms
        self.image_size = image_size
        self.patch_size= patch_size
        self.stride = stride
        self.color = color
        self.rescale = rescale

        # Initialize MNIST dataset
        self.seismic_handler = SeismicHandler()
        self.n_samples = self.seismic_handler.get_n_samples(subset) // terms
        self.n_batches = self.n_samples // batch_size
        
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.n_batches

    def next(self):

        # get batch of [batch_size*grid, image_size, image_size, 3]
        x, _ = self.seismic_handler.get_batch(self.subset, self.batch_size, self.image_size, self.patch_size, self.stride, self.color, self.rescale)
    
        # if stride is not the same as patch dim, then we assume it is halve of dim and have to subtract 1 from number of patches 
        if(int(self.patch_size/self.stride) ==1): n = int(self.image_size/self.stride)# -1)
        else: n = int(self.image_size/self.stride) -1
        nn = n*n

        # extract the rows to predict z+x
        # allocate a array with each colum [column number, rows, patch_size, patch_size, channels]
        sequence = np.empty([self.batch_size*n, n, x.shape[1], x.shape[2], x.shape[3]]) # TODO DYNAMIC CCOLUMN SIZE NOT 7
        # loop though each batch 
        counter = 0
        for i in range(self.batch_size):
            # and for each batch we extract every column
            for j in range(n):
                
                # print extracted column
                sequence[counter]= x[i*nn+j:i*nn+nn +j:n]
                counter +=1

      
        # generate negative predictive terms indexes 
        K = self.predict_terms
        # for random row
        nl = []
        nrr = []
        nrri = []
        for i in range(K):
            nlist = np.arange(0, n)
            nlist = nlist[nlist != (n-K+i)]
            nl.append(nlist)
            nrr.append([sorted(nl[i], key=lambda k: random.random()) for j in range(self.batch_size*n)])
        nrri = [np.stack([nrr[j][i][0] for j in range(K)], axis=0) for i in range(self.batch_size*n)]
        


        Y = [] # the actual predictive terms 
        Y_label = np.zeros((self.batch_size*n), dtype=np.float32) # the label of a predictive term, and hence for the entire column/sequence
        n_p = self.batch_size*n // 2
        for i in range(self.batch_size*n):
            if i <= n_p:
                Y.append(sequence[i, -K:, ...])
                Y_label[i] = 1
            else:
                 Y.append(sequence[i, nrri[i]])
        Y = np.asarray(Y)
        
        # verify by stacking columns to form a image 
        def stich_rows(n, terms, preds):
            fig = plt.figure(constrained_layout=True)
            widths = [1]*n
            heights = [1]*n
            spec5 = fig.add_gridspec(ncols=n, nrows=n, width_ratios=widths,
                                    height_ratios=heights)
            for row in range(n):
                for col in range(n):
                    ax = fig.add_subplot(spec5[row, col])
                    
                    #plot overlay for the terms to display label 
                    if row>=self.terms:
                        plt.imshow(preds[col][row-self.terms].astype('uint8'))
                        plt.imshow(np.ones(sequence[col][-self.terms].shape)*255, alpha=0.5)
                    else:
                        plt.imshow(terms[col][row].astype('uint8'))

                    #label = 'Width: {}\nHeight: {}'.format(widths[col], heights[row])
                    #ax.annotate(label, (0.1, 0.5), xycoords='axes fraction', va='center')
                    plt.axis('off')
            plt.show()


        # Assemble batch
        x_images = sequence[:, :-self.predict_terms, ...]

        # set random labels TODO do i need it 
        sentence_labels = np.ones((self.batch_size*n, 1)).astype('int32')
        # Randomize
        #idxs = np.random.choice(sentence_labels.shape[0], sentence_labels.shape[0], replace=False)
        return [x_images, Y], Y_label #[x_images[idxs, ...], y_images[idxs, ...]], sentence_labels[idxs, ...]


def plot_patches(patches):
    num_y, num_x =7,7
    ''' Draws a plot to stich together patches to a image '''
    counter = 1
    for y in range(num_y):
        for x in range(num_x):
            plt.subplot(num_y, num_x, counter)
            plt.imshow(patches[counter-1])
            plt.axis('off')
            counter += 1
    plt.show()

def plot_patches_column(patches, terms):
    n = patches.shape[0]

    ''' Draws a plot to stich together patches in one column '''
    counter = 1
    for i in range(n): 
        plt.subplot(n, 1, counter)
        plt.imshow(patches[i])
        if i >= terms:
            plt.imshow(patches[i],cmap=plt.cm.RdBu)
        else:
            plt.imshow(patches[i])
        plt.axis('off')
        counter += 1
    plt.show()




    


class SimilarityImageGenerator(object):

    ''' Data generator taking in 2d projection of images, sort them by their 2d distance and yield the most differetn from the ones already picked'''

    def __init__(self, ref, direction, treshold, unsorted):
    
        
        # Set params
        self.indx = 0
        self.ref = ref
        self.direction = direction
        self.treshold = treshold
        self.unsorted = unsorted
        self.ordered = self.order_by_dist(self.unsorted)#self.order_by_cluster_first(self.unsorted) 
        
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return len(self.ordered)

    def next(self):
        if len(self.ordered)>0:
            # calculate distance in slices to the next in the ordered list
            slice_dists = [abs(r[-1] - self.ordered[0][-1]) for r in self.ref]
            # continue to iterate until we the top ordered is outside threshold
            while any(dist <= self.treshold for dist in slice_dists):
                # remove if to close to a ref 
                del  self.ordered[0]
                # compute new dist to the nw top
                slice_dists = [abs(r[-1] - self.ordered[0][-1]) for r in self.ref]
            #delete it fomr the ordered list so it is not yielded again and add it to ref
            newRef = self.ordered.pop(0)
            self.ref.append(newRef)
            # then we need to reorder the list based on the new ref
            self.ordered = self.order_by_dist(self.ordered)
            return newRef
        else:
          raise StopIteration()
        
    def order_by_cluster_first(self, unsorted): 
        # we allocate a data variable to store our ordered data
        # this will be a list ordered first by cluster, then by eucledian distance 
        ordered = []
        # then we loop through each cluster and yeld the first image that is closes to the ref, but maintainig minLisceIndex away 
        for i in cluster_order[dir]:
            inter_cluster_points =unsorted[unsorted[:,2] == i, :]
            # then we sort these inter-cluster points withing the current cluster
            inter_cluster_points = inter_cluster_points.tolist()
            inter_cluster_points.sort(key=self.euclidean) # 3 column wsorted on ecleduan dist from ref (x, y, slice index)
            ordered = ordered + inter_cluster_points
        return ordered
    
    def order_by_dist(self, unsorted): 
        ordered = unsorted
        if type(unsorted) is not list:
            ordered = unsorted.tolist()
        ordered.sort(reverse=True, key=self.euclidean)
        return ordered
    
    def euclidean(self, coords):
        # we want the sum of the distance to all ref to be ass small/big as possible (depeing on finding similar/dissimilar)
        dist = []
        for coord in self.ref:
            xx, yy,_,_ = coord
            x, y,_,_ = coords
            dist.append(((x-xx)**2 + (y-yy)**2)**0.5)
        # calc  mean
        avg = sum(dist)/len(dist)
        # for each ref. calc deviatino form mean and sum up
        #dev = 0
        #for d in dist:
        #    dev = abs(d-avg)
        return sum(dist)

    

def generate_directional_dataset(data_cube):
    # define directions 
    directions = ["inline", "xline", "tline"]
    inline_samples, xline_samples, tline_samples = data_cube.shape
    data = {}
    labels = {}
    num_samples = {}
    shape={}
    
    num_samples["inline"] = inline_samples
    data["inline"] = data_cube
    labels["inline"] = np.arange(0,inline_samples)
    shape["inline"] = [xline_samples, tline_samples]

    num_samples["xline"] = xline_samples
    data["xline"] = np.transpose(data_cube, (1,0,2))
    labels["xline"] = np.arange(0,xline_samples)
    shape["xline"] = [inline_samples, tline_samples]


    num_samples["tline"] = tline_samples
    data["tline"] = np.transpose(data_cube, (2,1,0))
    labels["tline"] = np.arange(0,tline_samples)
    shape["tline"] = [inline_samples, xline_samples]
    return directions, inline_samples, xline_samples, tline_samples,data, labels, num_samples, shape


def flatten_directional_data(data, num_samples):
    data["inline"] = data["inline"].reshape(num_samples["inline"], -1)
    data["xline"] = data["xline"].reshape(num_samples["xline"], -1)
    data["tline"] = data["tline"].reshape(num_samples["tline"], -1)
    return data

def crop_image( img,  num_crops, crop_size):
        """ Crop image to a crop_size*crop_size crop, num_crop times"""
        crop = image.extract_patches_2d( img, (crop_size, crop_size), num_crops)
        return crop
