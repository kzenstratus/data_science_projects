from PIL import Image
import glob
from cascade import apply_weak_classifier
import numpy as np
from adaboost import calc_classifiers
from scipy import misc

def detect():
  ''' Given a classifier, return a set of locations
      from which we will draw on our final image
      strong = (h, curr_fpr, curr_fnr, big_theta)
      h = (alpha_t, h_t, featureIdx , polar, thresh)


  '''
  cascade = np.load("./data/cascades/cascade_2000.npy")
  feat_index = np.load("./data/features/feat_index.npy")
  test_idx, test_imgs = trim()
  
 

  for strong in cascade:
    h         = strong[0]
    big_theta = strong[3]
    h_test    = [] # all rounds of boosting
    h_single  = np.zeros(len(test_imgs)) # One round of boosting

    for weak in h:
      alpha_t   = weak[0]
      thresh    = weak[4]
      parity    = weak[3]
      featureIdx= weak[2]
      feature   = feat_index[featureIdx]

      for i in range(len(test_imgs)):
        img_gray = test_imgs[i]
        h_single[i] = apply_weak_classifier(img_gray,thresh,parity,feature)

      h_test.append(tuple( (alpha_t, h_single) ))
    signed_test = calc_classifiers(h_test,big_theta)

    # remove all the zeros in the signed_test
    remove_idx = []
    for j in range(len(signed_test)):
      if signed_test[j] == 0 :
        remove_idx.append(j)

    # remove images and corresponding indexes
    test_imgs = np.delete(test_imgs, remove_idx, axis = 0)
    test_idx  = np.delete(test_idx, remove_idx, axis = 0)


  draw("out.png",test_idx)


#def post_process(positions):
#  ''' Jank Rules:
#      Remove yourself if nothing within 2 pixels top right
#
#    Clean up multiple squares
#  '''
#  positions = sorted(positions, key=lambda x: x[0]+x[1], reverse = True)
#  clean_1 = []
#  for i in range(1,len(positions)):
#    left_x0 = positions[i-1][0]
#    top_x0 = positions[i-1][1]
#    left_x1 = positions[i][0]
#    top_x1 = positions[i][1]
#
#    if abs(left_x0-left_x1) + abs(top_x0-top_x1) <100:
#      clean_1.append(positions[i])
#  return clean_1





  
  

def draw(filename, location = [(600,400)]):
#    filename = "class.jpg"
    from PIL import Image, ImageDraw
    img = Image.open(filename)
    img_w, img_h = img.size
    background = Image.new('RGBA', (img_w, img_h), (255, 0, 0, 255))
    # background.paste(img, (0,0))

    window = 74
    img1 = Image.new('RGBA', (window, window))
    drw = ImageDraw.Draw(img1, 'RGBA') 
    drw.rectangle([(1,1), (window-1, window-1)], fill=None, outline=(255, 0, 0, 255))
    layer = Image.new('RGBA', (img_w, img_h))
    for loc in location:
        layer.paste(img1, tuple(loc))
    result = Image.composite(background, img, layer)
    # background.save('out.png')
    result.save('out.png')


def trim(Path = "./data/test/",filename = "./data/out.png",h = 64,w = 64):

    im = Image.open(filename).convert('LA')

    img_w, img_h = im.size

    test_idx  = []
    test_imgs = []
    for i in range(0,img_w, 5):
        for j in range(0,img_w,5):
            box = (j, i, j+ w, i+ h)
            test_idx.append((box[0],box[1]))
            temp = im.crop(box)
            temp.save("temp.png")
            temp = misc.imread("temp.png", flatten = 1)
            test_imgs.append(temp)
    np.save("./data/test/test_idx_5",test_idx)
    np.save("./data/test/test_imgs_5",test_imgs)
    return tuple((test_idx, test_imgs))
