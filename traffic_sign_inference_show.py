
import tensorflow as tf
import matplotlib.pyplot as plt

W=320
H=320
file1 ="./construction.jpg"
file2 ="./stop.jpg"
file3 ="./xpoodle.png"

file_contents = tf.read_file(file3)
im = tf.image.decode_jpeg(file_contents)
im  = tf.image.resize_image_with_crop_or_pad(im,256,256 )

im_bi  = tf.image.resize_images(im, (H,W), method=tf.image.ResizeMethod.BILINEAR)
im_nn  = tf.image.resize_images(im, (H,W), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
im_bic = tf.image.resize_images(im, (H,W), method=tf.image.ResizeMethod.BICUBIC)
im_ar  = tf.image.resize_images(im, (H,W), method=tf.image.ResizeMethod.AREA)

im_bi = tf.cast(im_bi, tf.uint8)
im_bic = tf.cast(im_bic, tf.uint8)
im_ar = tf.cast(im_ar, tf.uint8)

##im = tf.reshape(im, shape=[H,W, 3])

sess = tf.Session()
sess.run(tf.initialize_all_variables())

img_bi, img_nn, img_bic,img_ar = sess.run([im_bi, im_nn, im_bic, im_ar])

plt.imshow(img_bi)
plt.title("BILINEAR")
plt.figure()

plt.imshow(img_nn)
plt.title("NEAREST_NEIGHBOR")
plt.figure()

plt.imshow(img_bic)
plt.title("BICUBIC")
plt.figure()

plt.imshow(img_ar)
plt.title('AREA')
plt.show()

