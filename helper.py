import numpy as np
import cv2

def make_pie(image, repeater):
    image = np.expand_dims(image, 2)
    tp = tuple()
    for i in range(repeater):
        tp += tuple([image])
    pie = np.concatenate(tp, 2)
    return pie

def add_cream(pie, image):
    pie = np.roll(pie, 1, axis=2)
    pie[:,:,0] = image
    return pie

def convert_frame(image, sze=(84, 84)):
    image = cv2.resize(image, sze)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

def coin_toss(flappy = 0.5):
    return 0 if np.random.random(1) <= flappy else 1

def debug():
    img0 = np.random.random((2, 3))
    img1= make_pie(img0, 4)
    print img1
    print '\n\n\n'
    img2 = add_cream(img1, np.random.random((2, 3)))
    print img1, '\n'
    print img2
    print '\n\n\n\n'
    img3 = add_cream(img2, np.random.random((2, 3)))
    print img1, '\n'
    print img2, '\n'
    print img3
    print '\n\n\n\n\n'
    img4 = add_cream(img3, np.random.random((2, 3)))
    print img1, '\n'
    print img2, '\n'
    print img3, '\n'
    print img4
    print '\n\n\n\n\n\n'
    img5 = add_cream(img4, np.random.random((2, 3)))
    print img1, '\n'
    print img2, '\n'
    print img3, '\n'
    print img4, '\n'
    print img5
    print '\n\n\n\n\n\n\n'

def debug2():
    img0 = np.random.random((2, 3))
    img1= make_pie(img0, 4)
    img2 = img3 = img4 = img5 = img1
    print img1
    print '\n\n\n'
    img2 = add_cream(img1, np.random.random((2, 3)))
    print img1, '\n'
    print img2
    print '\n\n\n\n'
    img3 = add_cream(img2, np.random.random((2, 3)))
    print img1, '\n'
    print img2, '\n'
    print img3
    print '\n\n\n\n\n'
    img4 = add_cream(img3, np.random.random((2, 3)))
    print img1, '\n'
    print img2, '\n'
    print img3, '\n'
    print img4
    print '\n\n\n\n\n\n'
    img5 = add_cream(img4, np.random.random((2, 3)))
    print img1, '\n'
    print img2, '\n'
    print img3, '\n'
    print img4, '\n'
    print img5
    print '\n\n\n\n\n\n\n'


#if __name__ == '__main__':
    #debug()
    #debug2()
