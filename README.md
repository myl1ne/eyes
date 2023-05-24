# Background Swipe
This app aims at automatically segmenting the foregroud/backgroud in two images and swapping them both.

# Pipeline Logic
The pipeline is composed of multiple steps:

## Background/Foreground segmentation
I used SegmentAnything (SAM) to perform a semantic segmentation of elements in the images.
The background mask is then isolated through an heuristic by looking for a mask that touches both side of an image.
It is a simple heuristic yet it works well in the case of product shots which tend to be centered and rarely occupy a full side of an image.
Once the background mask is segmented, its inverse gives us the foreground.
We repeat this process on both images.

## Background holes inpainting
Once the foreground has been removed from the image it leaves a hole in the background, which will not necessarily have the shape of the other's image foreground.
For this reason we need to fill this hole before pasting the new foreground object.

Two methods are available in the pipeline:
### OpenCV Inpainting
Opencv proposes a simple method to inpaint based on color. It is fast and lightweight, but with quality that is far from photorealistic.
![Burger Processing OpenCV](/static/images/gitcontent/BurgerProcessing.png)
![Lipstick Processing OpenCV](/static/images/gitcontent/LipstickProcessing.png)

### StableDiffusion
SD offers an inpainting pipeline that is prompt based which can hallucinate good visual features (e.g continuing a pattern, adding details...), the tradeoff is for speed/RAM usage. A major drawback I faced is the model hallucinating back the foreground content based on the shape to be inpainted. See examples of this problem (the burger is fully reconstructed, which almost seems buggy, and bottles are added in the lipstick image):
![Burger Processing Stable Diff](/static/images/gitcontent/BurgerProcessingSD.png)
![Lipstick Processing Stable Diff](/static/images/gitcontent/LipstickProcessingSD.png)

### BIG Lama (NOT IMPLEMENTED)
BIG Lama is another inpainting model which could be a viable solution. I considered using it but ended-up short on time.
See: https://github.com/advimman/lama

## Resize Foreground to fit the other's background ratio
The last step is to actually "paste" the foreground on the other background.
Since images may not have the same resolution, it is needed to resize the foreground (expand or shrink) so that its bigger side matches one of the corresponding dimension of the target background.
During this process we ensure to keep the ratio of the foreground to avoid distortion of the product. We also center it in the image by adding padding where needed.
![Swapping Results](/static/images/gitcontent/SwappedBurgerLipstick.png)

# Packaging
You can test the pipeline through 2 ways:

* Interactive notebook:
A Jupyter notebook is provided in /notebook .

* Webapp
The app is packaged in a single class (BackgroundSwitcher.py) which is wrapped in a Flask server (app.py).
A simple UI (/templates/index.html) let's a user upload images and get results.
![Swapping Results](/static/images/gitcontent/WebAppShell.png)

# Installation
```
cd $REPO_ROOT
pip install -r requirements.txt
```

DL the SAM checkpoints and places them in $REPO_ROOT/checkpoints:
- (Large) https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
- (Medium) https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
- (Small) https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

Run the Flask server:
```
cd $REPO_ROOT
python app.py
```
Open your web browser to:
http://127.0.0.1:5000

If you turned on StableDiffusion, the first run will take a while as the server will download and cache the models checkpoint for the inpainting.


# Known issues:
- the contours of masks are sometimes too rough, even though the erode/dilate step I made to smooth it. One way to address the issue would be to alpha blend the peripheral regions of the foreground during the paste.
- there is sometime a bug in the padding process which causes and out of boundaries exception.

# Things tried:
## Cuing the foreground objects: 
SAM has 2 modes, an automated segmenter which parse the image into many masks (that is what I used) and a prompted one which takes a set of points or a box to give a region of interest to the model.
Since we were aiming at a fully automated process, I discarded the idea of having a human clicking, and instead tried to use a few methods to detect blobs (traditional opencv and torchvision with resnet).
In the end I found the heuristic about getting the background mask more robust than the automatic cuing method.
![Blob Detection](/static/images/gitcontent/BlobDetection.png)

# Room for improvement:
* Inpainting could be improved a lot. Lama seems promising, but other tweaks may improve the process. 
For example, we could "paste first, inpaint after": because we know that the foreground object will partially cover the hole, we could substract the foreground mask of the other image from the backround mask of the first, effectively reducing the area of the hole to inpaint. This could lead to much finer results with less noticeable distortions.
* Border smoothing: borders of the masks are not very well blended with the background. Instead of the "erode/dilate" method, one could actually create large borders on the masks and do inpainting there to better blend the elements.
* Artefacts removal: tiny disconnected masks sometimes still appear. There are thresholds to be tuned about the masks sizes that must be discarded.
* Benchmarking the system: I did not have time to implement an objective quality measure to see if the swaps were acceptable. I used subjective quality assessment. The most common visual issue after the swap is the sharpness of inclusion which could be measured automatically and provide an objective metric. Other metrics could be model based such as aesthetic scoring (e.g. the aesthetic scores of Image 1 and 2 should be roughly the same as image 3 and 4).