import pygame
import time
import numpy as np
import scipy as sp
from scipy import signal
from pygame.locals import *
def put_array(surface, myarr):          # put array into surface
    bv = surface.get_view("0")
    bv.write(myarr.tostring())
    del bv

def importance(arr):
	r_arr = (arr >> 16) % 256
	g_arr = (arr >> 8) % 256
	b_arr = arr % 256
	r_padded = np.lib.pad(r_arr, 1, 'reflect')
	g_padded = np.lib.pad(g_arr, 1, 'reflect')
	b_padded = np.lib.pad(b_arr, 1, 'reflect')
	return import_1chan(r_arr, r_padded) + import_1chan(g_arr, g_padded) + import_1chan(b_arr, b_padded)

def importance_sq(arr):
	r_arr = (arr >> 16) % 256
	g_arr = (arr >> 8) % 256
	b_arr = arr % 256
	r_padded = np.lib.pad(r_arr, 1, 'reflect')
	g_padded = np.lib.pad(g_arr, 1, 'reflect')
	b_padded = np.lib.pad(b_arr, 1, 'reflect')
	return import_1chan(r_arr, r_padded)**2 + import_1chan(g_arr, g_padded)**2 + import_1chan(b_arr, b_padded)**2

def importance_sq2(arr):
	r_arr = (arr >> 16) % 256
	g_arr = (arr >> 8) % 256
	b_arr = arr % 256
	r_padded = np.lib.pad(r_arr, 1, 'reflect')
	g_padded = np.lib.pad(g_arr, 1, 'reflect')
	b_padded = np.lib.pad(b_arr, 1, 'reflect')
	return sqimport_1chan(r_arr, r_padded) + sqimport_1chan(g_arr, g_padded) + sqimport_1chan(b_arr, b_padded)

def importance_sobel(arr):
	r_arr = (arr >> 16) % 256
	g_arr = (arr >> 8) % 256
	b_arr = arr % 256
	kern1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
	kern2 = kern1.T
	return (signal.convolve2d( r_arr, kern1, 'same')**2 + signal.convolve2d(g_arr, kern1, 'same')**2 + signal.convolve2d(b_arr, kern1,  'same')**2) \
	+ (signal.convolve2d( r_arr, kern2, 'same')**2 + signal.convolve2d(g_arr, kern2, 'same')**2 + signal.convolve2d( b_arr, kern2, 'same')**2) \

def importance_supersobel(arr):
	r_arr = (arr >> 16) % 256
	g_arr = (arr >> 8) % 256
	b_arr = arr % 256
	r_padded = np.lib.pad(r_arr, 3, 'reflect')
	g_padded = np.lib.pad(g_arr, 3, 'reflect')
	b_padded = np.lib.pad(b_arr, 3, 'reflect')
	kern1 = np.array(
	[[float(-3/18),float(-2/13),float(-1/10),float(0),float( 1/10),float(2/13),float(3/18)],
           [float(-3/13),float(-2/8),float( -1/5),float( 0),float( 1/5),float( 2/8),float( 3/13)],
           [float(-3/10),float(-2/5),float( -1/2),float( 0),float( 1/2),float( 2/5),float( 3/10)],
  [float(-3/9 ),float(-2/4),float( -1/1),float( 0),float( 1/1),float( 2/4),float( 3/9 )],
           [float(-3/10),float(-2/5),float( -1/2),float( 0),float( 1/2),float( 2/5),float( 3/10)],
           [float(-3/13),float(-2/8),float( -1/5),float( 0),float( 1/5),float( 2/8),float( 3/13)],
          [float(-3/18),float(-2/13),float(-1/10),float(0),float( 1/10),float(2/13),float(3/18)]])
	kern2 = kern1.T
	return (((signal.convolve2d( r_padded, kern1, 'same')**2 + signal.convolve2d(g_padded, kern1, 'same')**2 + signal.convolve2d(b_padded, kern1,  'same')**2) \
	+ (signal.convolve2d( r_padded, kern2, 'same')**2 + signal.convolve2d(g_padded, kern2, 'same')**2 + signal.convolve2d( b_padded, kern2, 'same')**2))[3:-3, 3:-3])*np.random.uniform(1, 1.2, r_arr.shape)


def importance_inv(arr):
	return -importance(arr)

def importance_invsq(arr):
	return -importance_sq(arr)

def importance_invsq2(arr):
	return -importance_sq2(arr)

def importance_invsobel(arr):
	return -importance_sobel(arr)

def importance_invsupersobel(arr):
	return -importance_supersobel(arr)

def importance_diag(arr):
	r_arr = (arr >> 16) % 256
	g_arr = (arr >> 8) % 256
	b_arr = arr % 256
	r_padded = np.lib.pad(r_arr, 1, 'reflect')
	g_padded = np.lib.pad(g_arr, 1, 'reflect')
	b_padded = np.lib.pad(b_arr, 1, 'reflect')
	return diagimport_1chan(r_arr, r_padded) + diagimport_1chan(g_arr, g_padded) + diagimport_1chan(b_arr, b_padded)

def import_1chan(arr, padded):
	return np.abs(arr - padded[1:-1, :-2]) + np.abs(arr - padded[1:-1, 2:]) + np.abs(arr - padded[:-2, 1:-1]) + np.abs(arr - padded[2:, 1:-1])

def sqimport_1chan(arr, padded):
	return (arr - padded[1:-1, :-2])**2 + (arr - padded[1:-1, 2:])**2 + (arr - padded[:-2, 1:-1])**2 + (arr - padded[2:, 1:-1])**2

def diagimport_1chan(arr, padded):
	return import_1chan(arr, padded) + np.abs(arr - padded[:-2, :-2]) + np.abs(arr - padded[2:, 2:]) + np.abs(arr - padded[:-2, 2:]) + np.abs(arr - padded[2:, :-2])

def abs_diff(c1, c2):
	return abs((c1 >> 16) - (c2 >> 16)) + abs((c1 >> 8) % 256 - (c2 >> 8) % 256) + abs(c1 % 65536 - c2 % 65536)

def getVerticalSeams(imp_arr):
	seam_arr = np.zeros_like(imp_arr)
	seam_arr[0, :] = imp_arr[0, :]
	for row in range(1, seam_arr.shape[0]):
		opt = np.minimum(imp_arr[row-1, 2:], np.minimum(imp_arr[row-1, :-2], imp_arr[row-1, 1:-1]))
		seam_arr[row, 1:-1] = imp_arr[row, 1:-1] + opt
		seam_arr[row, 0] = imp_arr[row, 0] + np.min(seam_arr[row-1, :2] )
        seam_arr[row, -1] = imp_arr[row, -1] + np.min(seam_arr[row-1, -2:] )
	return seam_arr

def getHorizontalSeams(imp_arr):
	seam_arr = np.zeros_like(imp_arr)
	seam_arr[:, 0] = imp_arr[:, 0]
	for col in range(1, seam_arr.shape[1]):
		opt = np.minimum(imp_arr[2:, col-1], np.minimum(imp_arr[:-2, col-1], imp_arr[1:-1, col-1]))
		seam_arr[1:-1, col] = imp_arr[1:-1, col] + opt
		seam_arr[0, col] = imp_arr[0, col] + np.min(seam_arr[:2, col-1] )
        seam_arr[-1, col] = imp_arr[-1, col] + np.min(seam_arr[-2:, col-1] )
	return seam_arr

In=1
pygame.init()
w = 768
h = 1024
size=[w,h]
screen = pygame.display.set_mode(size) 
screen.fill((0,0,0))
pygame.display.flip()
img=pygame.image.load('papawords.jpg')
img = pygame.transform.scale(img, size)
w = img.get_width()
h = img.get_height()
v = img.get_buffer()
temp1 = np.empty(w*h*3, dtype=int)
np.copyto(temp1,v)
temp2 = np.empty(w*h, dtype=int)
for i in range(w*h):
	temp2[i] = (temp1[3*i] << 16) + (temp1[3*i+1] << 8) + temp1[3*i+2]
pixels = temp2.reshape((h, w))
del v
del temp1
del temp2

x = y = 0
print len(pixels), len(pixels[0])
rowmask = np.ones(len(pixels), dtype=bool)
colmask = np.ones(len(pixels[0]), dtype=bool)
#rowmask[100:200] = False
#colmask[500:550] = False
rowsgone = 0
colsgone = 0
zeropad = np.zeros((h - rowsgone, colsgone), dtype=int)
zrow = np.zeros((h - rowsgone, 1), dtype=int)
print (w-rowsgone)*(h-colsgone)
print (pixels[rowmask, ...][..., colmask].shape[0]) * pixels[rowmask, ...][..., colmask].shape[1]
#print pixels[rowmask, ...][..., colmask].shape, zeropad.shape, np.append(pixels[rowmask, ...][..., colmask], zeropad, 0).shape
#print np.append(pixels[rowmask, ...][..., colmask], zeropad, 0)
vseams = []
hseams = []
while 1:
	pygame.event.pump()
	screen.fill((0,0,0))
	#print np.rot90(np.concatenate((pixels[rowmask, ...][..., colmask], zeropad), 0))
	# print pixels[rowmask, ...][..., colmask].shape, zeropad.shape
	#print pixels.shape, zeropad.shape
	put_array(screen, (np.concatenate((pixels, zeropad), 1)))
	pygame.display.flip()

	# processing
	#print 'starting impport'
	imp = importance_invsupersobel(pixels)
	#print imp
	#print 'ending import'
	# rows first
	seam_arr = getVerticalSeams(imp)
	#print 'after get v seams, seam arr:', seam_arr
	seam_start = x = np.argmin(seam_arr[0])
	y = 0
	seam = []
	cols = []
	while y < seam_arr.shape[0]:
		seam.append(x)
		cols.append(pixels[y, x])
		if y >= seam_arr.shape[0] - 1:
			break
		if x > 0:
			x += np.argmin(seam_arr[y+1, x-1:x+2]) - 1
		else:
			x += np.argmin(seam_arr[y+1, x:x+2])
		y += 1
		#print dir, x, y


	vseams.append((seam, cols))
	#print 'before all: ', pixels.shape, minpath_arr, 'seam len is', len(seam), 'y is', y
	pixels = np.array([np.delete(pixels[row], seam[row], axis=0) for row in range(len(seam))])
	#print 'after v: ', pixels.shape
	# processing
	#print 'starting impport'
	imp = importance_invsupersobel(pixels)
	#print imp
	#print 'ending import'
	# rows first
	seam_arr = getHorizontalSeams(imp)
	#print 'after get h seams, seam arr:', seam_arr
	seam_start = y = np.argmin(seam_arr[:, 0])
	x = 0
	seam = []
	cols = []
	while x < seam_arr.shape[1]: #and dir != -2:
		seam.append(y)
		cols.append(pixels[y, x])
		if x >= seam_arr.shape[1] - 1:
			break
		if y > 0:
			y += np.argmin(seam_arr[y-1:y+2, x+1]) - 1
		else:
			y += np.argmin(seam_arr[y:y+2, x+1])
		x += 1
		#print dir, x, y

	hseams.append((seam, cols))
	pixels = np.array([np.delete(pixels[:, col], seam[col], axis=0) for col in range(len(seam))]).T
	#print 'after h:', pixels.shape, 'seam len is', len(seam)
	#print seam_arr
	#print zeropad.shape
	zeropad = np.delete(zeropad, -1, 0)
	#print zeropad.shape
	zrow = np.delete(zrow, -1, 0)
	zeropad = np.append(zeropad, zrow, 1)
	#rowmask[200+x] = False
	#colmask[499-x] = False
	rowsgone+=1
	colsgone+=1
	print 'frame %d of %d done' % (rowsgone, 766)
	if rowsgone == 766:
		break
suffix = 'invsupersobelrand'
with open('vseams'+suffix, 'w') as f:
	f.write(repr(vseams))

with open('hseams'+suffix, 'w') as f:
	f.write(repr(hseams))

with open('startpix'+suffix, 'w') as f:
	f.write(repr(pixels))
