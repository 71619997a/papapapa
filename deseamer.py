import pygame
import time
import numpy as np
from numpy import array
from pygame.locals import *
def put_array(surface, myarr):          # put array into surface
    bv = surface.get_view("0")
    bv.write(myarr.tostring())
    del bv

def add_vseam(pixels, seam, cols):
	return np.array([np.insert(pixels[row], seam[row], cols[row]) for row in range(len(seam))])

def add_hseam(pixels, seam, cols):
	return np.array([np.insert(pixels[:, col], seam[col], cols[col]) for col in range(len(seam))]).T

In=1
suffix = 'inv'
pygame.init()
w = 768
h = 1024
size=[w,h]
screen = pygame.display.set_mode(size) 
screen.fill((0,0,0))
pygame.display.flip()

with open('startpix'+suffix) as f:
	pixels = eval(f.read())

x = y = 0
print len(pixels), len(pixels[0])
rowmask = np.ones(len(pixels), dtype=bool)
colmask = np.ones(len(pixels[0]), dtype=bool)
#rowmask[100:200] = False
#colmask[500:550] = False
rowsgone = 766
colsgone = 766
zeropad = np.zeros((h - rowsgone, colsgone), dtype=int)
zcol = np.zeros((1, colsgone), dtype=int)
print (w-rowsgone)*(h-colsgone)
#print (pixels[rowmask, ...][..., colmask].shape[0]) * pixels[rowmask, ...][..., colmask].shape[1]
#print pixels[rowmask, ...][..., colmask].shape, zeropad.shape, np.append(pixels[rowmask, ...][..., colmask], zeropad, 0).shape
#print np.append(pixels[rowmask, ...][..., colmask], zeropad, 0)
with open('vseams'+suffix) as f:
	vseams = eval(f.read())
with open('hseams'+suffix) as f:
	hseams = eval(f.read())
R = (255, 0, 0)
hseam_prev = []
vseam_prev = []
numprevseams = 15
while 1:
	pygame.event.pump()
	st_t = time.time()
	#screen.fill((0,0,0))
	#print np.rot90(np.concatenate((pixels[rowmask, ...][..., colmask], zeropad), 0))
	# print pixels[rowmask, ...][..., colmask].shape, zeropad.shape
	print pixels.shape, zeropad.shape
	put_array(screen, (np.concatenate((pixels, zeropad), 1)))
	for i in range(len(hseam_prev)):
		hseam = hseam_prev[i]
		vseam = vseam_prev[i]
		col = (255 * (numprevseams - i - 1) / (numprevseams - 1), (255 * i) / (numprevseams - 1), 0)
		for x in range(len(hseam)):
			y = hseam[x]
			screen.set_at((x, y), col)
		for y in range(len(vseam)):
			x = vseam[y]
			screen.set_at((x, y), col)
	pygame.display.flip()
	hseam, hcols = hseams.pop()
	vseam, vcols = vseams.pop()
	pixels = add_hseam(pixels, hseam, hcols)
	pixels = add_vseam(pixels, vseam, vcols)
	if len(hseam_prev) >= numprevseams:
		hseam_prev.pop(0)
		vseam_prev.pop(0)
	hseam_prev.append(hseam)
	vseam_prev.append(vseam)
	#zcol = np.append(zcol, [[0]], 1)
	#zeropad = np.append(zeropad, zcol, 1)
	#print zeropad.shape
	#zeropad = np.delete(zeropad, -1, 0)
	#rowmask[200+x] = False
	#colmask[499-x] = False
	rowsgone-=1
	colsgone-=1
	zeropad = np.zeros((h - rowsgone, colsgone), dtype=int)
	print 'frame %d of %d undone' % (rowsgone, 766)
	time.sleep(max(0.05 - (time.time() - st_t), 0))
	if rowsgone == 0:
		time.sleep(2)
		put_array(screen, (np.concatenate((pixels, zeropad), 1)))
		pygame.display.flip()
		break
raw_input()