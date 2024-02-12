# What is this project about
This project, developed for my university couse of `Signal, Image and Video` focuses on verifying the possiblity to estimate the size of kiwi-fruits from 2D images taken directly on the field, using Machine Learning as little as possible.  
The application of this software could be beneficial for farmers that want to measure the performances of their land, this is just the prototype of what it could be possible to do, such as visualize the production quality identifying plant lines, orchards and, if the software is used by many farmers it could be useful to improve quality, production quantity and research about those.

# How it works
1. Camera distortion is corrected
2. Images are cropped around the fruit bins and are made square
3. Kiwi-fruit instances on the top of the bin are segmentated with Mask-RCNN
4. Masks are filtered so that only the best positioned fruits for size estimation are kept. This is done with many parameters such as:
    - instance segmentation prediction accuracy (direct Mask-RCNN output filter)
    - area occupied from the fruit - used to overcome the underfitting of the model
    - overlapping masks are removed
    - masks convexity is evaluated and those who are too convex are removed from the pool
    - if fruit-stem and/or flower-stem are present the mask is removed
5. Kiwi-fruits sizes are estimated in px from the closest ellipse approximable on their contours
6. px size is transformed to cm - the fixed measurement is the side of the plastic bin
7. volume is estimated as an elissoid with h=measures height, w1=w2=measured width - this is a big approximation as kiwi-fruits actually have two different Equatorial Widths
8. the mass is calculated from the volume
9. classification and plot

## Usage
This project uses [Mask-RCNN](https://arxiv.org/abs/1703.06870) for instance segmentation, a GPU capable of Cuda is required.

Download Mask-RCNN weights from [here](https://drive.google.com/file/d/1EWLDJOHahNgTNjPA7T4WSUwNMwCCbQkH/view?usp=drive_link). They should be put into:
> src/mask_rcnn/logs/kiwi20240123T1953

If you want to use the testing notebooks download precalculated masks from [here](https://drive.google.com/file/d/1uzYOISRta8dgY9tUD9zdWVspk5Hmf_AI/view?usp=drive_link). It should be put into:
> src/mask_rcnn 

To run this project use `python 3.10.12`
> pip install -r requirements.txt  
> python3 serv.py

The program will analyze images put in `raw_selected` and save `final_plot.png` showing the estimated fruit sizes from the top layers of the bins.

# Sources
- Image segmentation: [Mask-RCNN TF2.14 and Python 3.10.12](https://github.com/z-mahmud22/Mask-RCNN_TF2.14.0)
- [Fruit-Images-Dataset](https://github.com/Horea94/Fruit-Images-Dataset/tree/master) used for Mask-RCNN's training
- I captured images of kiwifruit bins locally for this project. These images were taken by me, and I've removed the EXIF data for privacy and security reasons.
