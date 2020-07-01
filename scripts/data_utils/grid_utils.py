import matplotlib
import os
#matplotlib.use('PS')
#matplotlib.use(
#    "TkAgg"
#)  # suggested to avoid AttributeError: 'FigureCanvasMac' object has no attribute 'renderer'
import matplotlib.pyplot as plt 
import numpy as np

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array looks like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
            .swapaxes(1,2)
            .reshape(-1, nrows, ncols))


def unblockshaped(arr, h, w, channels=False):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    
    if channels:
        n, nrows, ncols, channels = arr.shape
        return (arr.reshape(h//nrows, -1, nrows, ncols,channels)
                .swapaxes(1,2)
                .reshape(h, w,3))
    else:
        n, nrows, ncols = arr.shape
        return (arr.reshape(h//nrows, -1, nrows, ncols)
                .swapaxes(1,2)
                .reshape(h, w))




def plot_embeddings(patches, dim, stride, image_size=256, stitched=False, channels=None, color_patches=False):
    ''' Draws a plot of stiched together patches or indidividual patches '''
    counter = 1
    # orignal image is 256*256. This image is  448*448
    # e.g if we have 32 overlap we ofetch out every other images in both directions
    
    # if stride is not the same as patch dim, then we assume it is halve of dim and have to subtract 1 from number of patches 
    if(int(dim/stride) ==1): n = int(image_size/stride)# -1)
    else: n = int(image_size/stride) -1

    skip_every = int(dim/stride)
    num_dir_patches =int(image_size/dim)
    #stich togheter image from 7*7*64 to a full 265 image
    if stitched:
        indices = []
        if(num_dir_patches == 4):
            for i in range(num_dir_patches):
                indices.append((0*skip_every)+i*skip_every*n)
                indices.append((1*skip_every)+i*skip_every*n)
                indices.append((2*skip_every)+i*skip_every*n)
                indices.append((3*skip_every)+i*skip_every*n)
        elif (num_dir_patches == 8):
            for i in range(num_dir_patches):
                indices.append((0*skip_every)+i*skip_every*n)
                indices.append((1*skip_every)+i*skip_every*n)
                indices.append((2*skip_every)+i*skip_every*n)
                indices.append((3*skip_every)+i*skip_every*n)
                indices.append((4*skip_every)+i*skip_every*n)
                indices.append((5*skip_every)+i*skip_every*n)
                indices.append((6*skip_every)+i*skip_every*n)
                indices.append((7*skip_every)+i*skip_every*n)
        non_overlapping_patches = patches[np.array(indices)] 
        full_img =unblockshaped(non_overlapping_patches, 256, 256, channels=True)
        plt.imshow(full_img)#.astype('uint8'))
        # add a color grid to highlight the patches
        if color_patches:
            grid_overlay = np.copy(non_overlapping_patches)
            for i in range(grid_overlay.shape[0]):
                grid_overlay[i,:,:,0] = np.ones(grid_overlay[i,:,:,0].shape[-1])*(1/grid_overlay.shape[0]*i)
            grid_img =unblockshaped(grid_overlay, 256, 256, channels=True)
            plt.imshow(grid_img)#.astype('uint8'))
    else:
        for i in range(n*n):
            plt.subplot(n, n, counter)
            if channels:
                plt.imshow(patches[i].reshape(dim,dim,channels).astype('uint8'))
            else:
                plt.imshow(patches[i].reshape(dim,dim).astype('uint8'))
            plt.axis('off')
            counter += 1
    plt.show() 


def pad_data_to_fit_patch(data, patch_size, directions, num_samples, output_folder):
    padding_needed = {}
    for dir in directions:
        padding_needed[dir] = patch_size-(num_samples[dir]%patch_size)

    # perform padding on each directional data 
    padded_data = {}

    filename = output_folder + "/padded_inline_" + str(patch_size) + ".npy"
    if not os.path.exists(filename):
        print("Creating inline")
        padded_data["inline"] = np.pad(data["inline"], ((0, padding_needed["inline"]),(0, padding_needed["xline"]),(0, padding_needed["tline"])), 'constant', constant_values=(0,0))
        np.save(filename,padded_data["inline"])
    else: 
        print("Found inline")
        padded_data["inline"] = np.load(filename)

    filename = output_folder + "/padded_xline_" + str(patch_size) + ".npy"
    if not os.path.exists(filename):
        print("Creating xline")
        padded_data["xline"] = np.pad(data["xline"], ((0, padding_needed["xline"]),(0, padding_needed["inline"]),(0, padding_needed["tline"])), 'constant', constant_values=(0,0))
        np.save(filename,padded_data["xline"])
    else: 
        print("Found xline")
        padded_data["xline"] = np.load(filename)

    filename = output_folder + "/padded_tline_" + str(patch_size) + ".npy"
    if not os.path.exists(filename):
        print("Creating tline")
        padded_data["tline"] = np.pad(data["tline"], ((0, padding_needed["tline"]),(0, padding_needed["xline"]),(0, padding_needed["inline"])), 'constant', constant_values=(0,0))
        np.save(filename,padded_data["tline"])

    else: 
        print("Found tline")
        padded_data["tline"] = np.load(filename)
    return padded_data 

def create_patched_data(padded_data, patch_size, directions):
    # allocate array
    patched_data = {}
    patched_labels = {}
    patched_labels_per_image = {}
    patched_labels_per_grid_cell={}
    patched_grid_sizes = {}
    patched_num_smaples = {}

    for dir in directions:
        # for inline 
        shape = padded_data[dir].shape
        n = int(shape[1]/patch_size) 
        m = int(shape[2]/patch_size)
        tot_patches = shape[0]*n*m
        patched_num_smaples[dir] = tot_patches
        patched_grid_sizes[dir] = [tot_patches, n, m]
        patched_data[dir] = np.empty([tot_patches, patch_size, patch_size])
        patched_labels[dir] = np.empty([tot_patches])
        patched_labels_per_image[dir] = np.empty([tot_patches])
        patched_labels_per_grid_cell[dir] = np.empty([tot_patches])
        # we start in inline direction 
        for i in range(shape[0]):
            slice = padded_data[dir][i].T
            patched_data[dir][i*n*m:(i*n*m)+(n*m)] = blockshaped(slice, patch_size, patch_size)
            patched_labels[dir][i*n*m:(i*n*m)+(n*m)] = np.arange(i*n*m,(i*n*m)+(n*m))  # these are labels per patch.... 
            patched_labels_per_image[dir][i*n*m:(i*n*m)+(n*m)] = np.floor(np.arange(i*n*m,(i*n*m)+(n*m))/(n*m))  # these are labels per image, meaning all patches in a image get the same index.... 
            patched_labels_per_grid_cell[dir][i*n*m:(i*n*m)+(n*m)] = np.arange(i*n*m,(i*n*m)+(n*m))%(n*m)  # these are labels per grid cell, meaning that all have a label in rthe range (0, n*m)
    return patched_data, patched_labels, patched_labels_per_image, patched_labels_per_grid_cell, patched_labels_per_grid_cell, patched_grid_sizes, patched_num_smaples

def plot_patched_directions(padded_data, patched_data, patched_grid_sizes, patch_size, directions, output):
    for dir in directions:
        max = padded_data[dir].shape[0]
        n = patched_grid_sizes[dir][1]
        m = patched_grid_sizes[dir][2]
        nm = patched_grid_sizes[dir][1]*patched_grid_sizes[dir][2]
        # verify a random slice to see if patcheing works
        random_slice = np.random.choice(max, 1)[0]
        random_slice  = random_slice*nm
        patched_slice = patched_data[dir][random_slice:random_slice+nm]
        counter = 1
        for i in range(nm):
                plt.subplot(m,n, counter)
                if patched_slice.shape[-1]== 3:
                    plt.imshow(patched_slice[i].reshape(patch_size,patch_size,3).astype('uint8'))
                else:
                    plt.imshow(patched_slice[i].reshape(patch_size,patch_size).astype('uint8'))
                plt.axis('off')
                counter += 1
        if output:
            filename = output + "/example_patch_image" + dir +"_"+ str(patch_size) + ".png"
            plt.savefig(filename)
        else:
            plt.show()
        
        # plot_embeddings(patched_slice, patch_size, patch_size, image_size=image_size, stitched=True, channels=3)
        
