# NBP-DemoClient-User

A Work In Progress client for the number plate game. This is an adapted version of the [battleships bot](https://test.aigaming.com/Help?url=downloads) from AIGaming.com.

## Packages

This project makes use of several common libraries:

- [numpy](http://www.numpy.org)
- [Pillow](http://pillow.readthedocs.io) for image processing
- [matplotlib](https://matplotlib.org) only for jupyter notebooks
- [tkinter](https://wiki.python.org/moin/TkInter) for the UI
- [requests](http://docs.python-requests.org) to talk with the API server

There exist other powerful python packages for image processing which have not been used in this project in order to conserve future server space. They could however be used though the python API. These are:

- [cv2](https://docs.opencv.org/3.0-beta) which is currently in its _beta_ for python 3. Also; it's huge (large memory foorprint)
- [scikit-image](http://scikit-image.org) A well-made collection of algorithms for image processing. It has good documentation and examples.
- [scipy.ndimage](https://docs.scipy.org/doc/scipy/reference/ndimage.html) Multi-dimentional image processing part of scipy

## Resources

- University of Oxford robotics [image analysis notes](http://www.robots.ox.ac.uk/~az/lectures/ia/)
- [Grayscale image entropy](http://uk.mathworks.com/help/images/ref/entropy.html) - could be useful for something and it's pretty [easy to implement](http://snipplr.com/view/20341/image-entropy-from-pil-histogram/)
- [RDP algorithm](https://en.wikipedia.org/wiki/Ramer–Douglas–Peucker_algorithm) - algorithm for simplifying a piecewise linear curve
- [contour approximation](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html) in cv2
- Minimal Bag of Visual Words [Image Classifier](https://github.com/shackenberg/Minimal-Bag-of-Visual-Words-Image-Classifier)
- University of Washington [computer vision lecture slides](https://courses.cs.washington.edu/courses/cse455/09wi/Lects/)
- [Matplotlib colormap in a PIL image](https://stackoverflow.com/questions/10965417/how-to-convert-numpy-array-to-pil-image-applying-matplotlib-colormap)
- [Static variables](https://stackoverflow.com/questions/279561/what-is-the-python-equivalent-of-static-variables-inside-a-function) in python
- [Number plate font](https://www.dafont.com/uk-number-plate.font)
- [Scale-Invariant Feature Transform](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform) and [Speeded up robust features](https://en.wikipedia.org/wiki/Speeded_up_robust_features) are two feature-based algorithms (and a very good stack exchange [post](https://stackoverflow.com/questions/10168686/image-processing-algorithm-improvement-for-coca-cola-can-recognition?rq=1) about them)

## ToDo's

Things which need to be done and a few ideas on what could be done

### Easy

- Make use of sending off multiple moves in one request

### Medium

- Make use of some number plate statistics (for example [this](https://www.gov.uk/vehicle-registration/q-registration-numbers))
- Pull images from the links provided in the `gamestate` instead of using the hard-coded library.
- Make use of the color in the images. This will require one to rewrite a bunch of code in the helper functions including some really core stuff like `pil2np` and `np2pil` though
- 

### Hard

- Write own class for (RGB?) images which should probably hold them as `numpy` arrays and convert to `PIL` when necessary
- Make the `RMSMultiSearch` faster
- Use some cutting-edge cv2/scikit-learn trickery voodoo