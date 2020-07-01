import matplotlib
#matplotlib.use('PS')
#matplotlib.use(
#    "TkAgg"
#)  # suggested to avoid AttributeError: 'FigureCanvasMac' object has no attribute 'renderer'
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.patheffects as PathEffects
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# plot with/without text label 
def scatter(x, labels, num_labels, dir, output=False, show_labels=False, s=100):
    plt.figure(figsize=(15,10))
    ax = plt.subplot()
    # each image is projected to a x,y coordinate. 
    # scatter the different images with the original label 
    plt.scatter(x[:, 0], x[:, 1], lw=0.25, c=labels, edgecolor='k',  s=s, cmap=plt.cm.get_cmap('cubehelix', num_labels))
    plt.xlabel('PC1', size=20), plt.ylabel('PC2', size=20), plt.title("2D Projection of patches" + dir, size=25)
    plt.colorbar(ticks=np.arange(0,num_labels,num_labels/2), label='digit value')
    plt.clim(0,num_labels)
    
    # We add the labels for each cluster.
    txts = []
    labels = np.asarray(labels)
    if show_labels:
        for i in range(num_labels):
            # Position of each label.
            xtext, ytext = np.median(x[labels == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=20)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)
    if output: 
        plt.savefig(output)



def visualize_scatter_with_images(X_2d_data, images, figsize=(45,45), image_zoom=1, output=False):
    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for xy, i in zip(X_2d_data, images):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(X_2d_data)
    ax.autoscale()
    
    if output: 
        plt.savefig(output)

def plot_random_directions(data, max):
    # plt different slices 
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    random_slice = np.random.choice(max, 1)[0]
    ax1.imshow(data["inline"][random_slice].T)
    ax1.set_title('inline')

    ax2.imshow(data["xline"][random_slice].T)
    ax2.set_title('xline')

    ax3.imshow(data["tline"][random_slice].T)
    ax3.set_title('tline')

    plt.show()

def plot_subset_of_directions(data, directions, max):
    for dir in directions:
        j = 1
        np.random.seed(1)
        fig = plt.figure(figsize=(10,10)) 
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)  
        for i in np.random.choice(400, 9):
            # plot a 8*8 imges 25 times
            plt.subplot(3,3,j), plt.imshow(data[dir][i].T), plt.axis('off')
            j += 1
        plt.show()

def plot_similar_image(i, next_indx, next_img, ref_imgs, ref_data, proj, proj_labels, proj_num_samples):
    plt.imshow(next_img)
    plt.title(f'{i}th most different image at index {next_indx}')
    plt.show()
    
    # plot all refs as labels on top of 
    refs = np.asarray(ref_data)
    #print("refs", refs[:,:2],  refs[:,-1], refs.shape[0])
    plt.figure(figsize=(15,10))
    ax = plt.subplot()
    # each image is projected to a x,y coordinate. 
    # scatter the different images with the original label 

    plt.scatter(proj[:, 0], proj[:, 1], lw=0.25, c=proj_labels, edgecolor='k',  s=100, cmap=plt.cm.get_cmap('cubehelix', proj_num_samples))
    plt.xlabel('PC1', size=20), plt.ylabel('PC2', size=20), plt.title("2D Projection of patches ", size=25)
    plt.colorbar(ticks=np.arange(0,proj_num_samples,proj_num_samples/2), label='digit value')
    plt.clim(0,proj_num_samples)
    # add the reference points
    plt.scatter(ref_data[:-1, 0], ref_data[:-1, 1], lw=0.25,   s=500, color='red')
    # add the last reference point to highlight ned new additon
    plt.scatter(ref_data[-1, 0], ref_data[-1, 1], lw=0.25,   s=500, color='blue')
    #scatter(refs[:,:2],  refs[:,-1], n_clusters, direction, s=500, show_labels=False)
    plt.show() 

    
