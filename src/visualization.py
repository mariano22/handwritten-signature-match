import cv2
import PIL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Make a tensor ready to be displayed
def showable(t):
    t = t.detach().cpu()
    # it assumes it was normalized with Imagenet stats
    inverse_transform = transforms.Compose([
        transforms.Normalize(mean=[0,0,0], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1,1,1]),
    ])
    if t.min()<0:
        t = inverse_transform(t)
    if t.size(0) == 3:
        t = t.permute(1, 2, 0)
    return t

def show_tensor(tensor):
    tensor = showable(tensor)
    plt.imshow(tensor)
    plt.axis('off')  # Hide axis
    plt.show()

def show_pair(e):
    fig, axes = plt.subplots(1, 2, figsize=(10, 2), facecolor='white')
    for ax,x in zip(axes,e[:2]):
        if isinstance(x,str):
            img = plt.cm.gray(mpimg.imread(x))
        elif isinstance(x,torch.Tensor):
            img = showable(x)
        else:
            img = x
        ax.imshow(img)
        ax.axis('off')  # Hide axes for a cleaner look
    plt.suptitle(['Mismatch','Match'][e[2]==e[3]], fontsize=16)
    plt.tight_layout()
    plt.show() 

def show_grid(images, subtitles, figsize):
    rows = len(images)
    cols = len(images[0])
    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize[0], rows * figsize[1]))
    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(showable(images[i][j]), cmap='gray')
            axes[i, j].set_title(subtitles[i][j], fontsize=10)
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()
    return fig

def show_pair_batch(x1 , x2, y1, y2, columns=8):
    batch_size, channel_size, height, width = x1.shape
    assert batch_size % columns == 0, f"Columns ({columns}) must be a divisor of batch_size ({batch_size})"
    images = torch.stack([x1, x2], dim=2).view(batch_size, channel_size, height*2, width)
    images = images.view(-1, columns, channel_size, height*2, width)
    labels = (y1==y2)
    subtitles = [ [ ['Missmatch', 'Match'][x] for x in xs] for xs in labels.view(-1,columns).tolist() ]
    show_grid(images, subtitles, (2,2))

def show_batch_images(x, images_per_row=8):
    B,C,H,W = x.shape
    show_tensor(x.view(-1,images_per_row,C,H,W).permute(2,0,3,1,4).reshape(C,H*B//images_per_row,W*images_per_row))

def mark_img(img, x, y,w=7, c=0):
    """ 
    Draw a point mark of w width in (x,y) in the image choosing color c.
    Example:
    ds = sigcomp2009()
    timg = torchvision.io.read_image(ds[0][0])
    img = TF.to_pil_image(timg).convert('RGB')
    mark_img(img,1713//2,530//2,w=20) 
    """
    colors =['red', 'blue', 'green', 'yellow']
    X, Y = img.size
    x0 = max(0,x-w)
    y0 = max(0,y-w)
    x1 = min(X,x+w)
    y1 = min(Y,y+w)
    ImageDraw.Draw(img).rectangle([x0,y0,x1,y1],fill=colors[c])

def mark_img_line(img, p0, p1, w=7, c=0):
    """ 
    Draw a line from p0 to p1 of w width in the image choosing color c.  
    """
    colors =['red', 'blue', 'green', 'yellow']
    X, Y = img.size
    ImageDraw.Draw(img).line([p0[0],p0[1],p1[0],p1[1]],fill=colors[c],width=w)


def preprocess_choose(img):
    """ Choose between different preprocess methods. """
    npimg = np.array(img)
    _,otsu = cv2.threshold(npimg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _,otsugauss = cv2.threshold(cv2.GaussianBlur(npimg,(5,5),0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _,fixth = cv2.threshold(npimg,127,255,cv2.THRESH_BINARY)
    adap = cv2.adaptiveThreshold(npimg,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    adapgauss = cv2.adaptiveThreshold(npimg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    images = [ [otsu, otsugauss], 
               [fixth, adap], 
               [adapgauss, img] ]

    titles = [ ['Otsu', 'Otsu + Gaussian Blur'], 
               ['Global Thresholding (v = 127)', 'Adaptive Mean Thresholding'], 
               ['Adaptive Gaussian Thresholding', 'Original Image'] ]
    return show_grid( images, titles, figsize=(6,2) )

def plot_2d_space(X, y, title='Classes'):
    """ Plot a 2D 2-class dataset X (2D), y (2 values only). """
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        #if m == 's':
        #    continue
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(title)
    plt.legend(loc='upper right')
    plt.show()


def plotpca2d(xs_pca, ys, cls_start=None, n_cls=None, title=None):
    assert (cls_start==None) == (n_cls==None)
    if cls_start is None:
        cls_start = 0
        n_cls = len(set(ys.tolist()))
    fig = plt.figure()
    plt.title('Embeddings')

    cm = plt.get_cmap('gist_rainbow')
    ax =  plt.axes()
    ax.set_prop_cycle('color', [cm(1.*i/n_cls) for i in range(n_cls)])

    for i in range(cls_start,cls_start+n_cls):
        plt.plot(xs_pca[ys==i,0], xs_pca[ys==i,1], '.', label=f'class {i}')
    if title: plt.title(title)

def plotpca3d(xs_pca, ys, cls_start=None, n_cls=None, title=None):
    assert (cls_start==None) == (n_cls==None)
    if cls_start is None:
        cls_start = 0
        n_cls = len(set(ys.tolist()))
    fig = plt.figure(figsize=(10,10))
    plt.title('Embeddings')

    cm = plt.get_cmap('gist_rainbow')
    ax = plt.axes(projection='3d')
    ax.set_prop_cycle('color', [cm(1.*i/n_cls) for i in range(n_cls)])

    for i in range(cls_start,cls_start+n_cls):
        ax.plot3D(xs_pca[ys==i,0], xs_pca[ys==i,1], xs_pca[ys==i,2], '.', label=f'class {i}')
    if title: plt.title(title)
    plt.legend(loc="lower left")

def plot_precision_recall(precs, recs, metric_precision, metric_recall, figuresize=(12,8),title=None):
    plt.rcParams["figure.figsize"] = figuresize
    disp = PrecisionRecallDisplay(precision=precs, recall=recs)
    disp.plot()
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    plt.axvline(metric_recall, color='r')
    plt.axhline(metric_precision, color='b')
    print(f'Recall={metric_recall} | Precision={metric_precision}')
    
    if title:
        plt.title(title)
    plt.show()

def plot_prec_rec_by_th(th,rec,prec,f1,trained_th=None,figuresize=(12,8),title=None):
    plt.rcParams["figure.figsize"] = figuresize
    if trained_th:
        plt.axvline(trained_th, color='k')
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.xlabel('Decision function threshold')
    plt.scatter(th,f1[:-1],color='y', label='f1')
    plt.scatter(th,rec[:-1],color='r', label='recall')
    plt.scatter(th,prec[:-1],color='b', label='precision')
    plt.legend(loc="lower left")
    if title:
        plt.title(title)

def plots_prec_rec(ns,th,figuresize=(12,8)):
    plot_prec_rec_by_th(ns.ths,ns.recs,ns.precs,ns.f1s,th,figuresize,title=ns.title)
    plot_precision_recall(ns.recs, ns.precs,ns.metrics[0], ns.metrics[1],figuresize,title=ns.title)