import os, sys
import numpy as np
import pylab as pl
import matplotlib.cm as cm
import matplotlib.pyplot as pp
import matplotlib.image as img
import Image, ImageDraw as id

max_depth = 60
max_iteration = 5
keep_running = 1
#SAFE POINT TABLE 0-255
safe_point_lookup = [0, 2, 0, 2, 2, 0, 2, 2, 0, 2, 0, 1, 0, 0, 2, 1, 0, 0, 0, 2, 2, 0, 1, 1, 0, 0, 0, 2, 0, 0, 2, 1,
					 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
					 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,
					 2, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 2, 0, 0, 0, 2, 1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
					 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
					 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
					 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0,
					 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 0, 0, 0, 2, 1, 0, 0, 0, 1, 0, 1, 1, 2, 2, 0, 0, 2, 2, 0, 0]

def quad():
	im = Image.open("black_top.png")
	im = im.convert("1")
	#buffer for printing out
	im2 = Image.new("1", im.size, 255)
	#check image color
	im_hist = im.histogram()
	im_color = 1
	if(im_hist[0]>0 and im_hist[255]>0):
		im_color = 2
	elif(im_hist[0]>0):
		im_color = 0
	#build quad tree, still hardcoding the init to gray
	
	#if there is still point to be removed
	iteration = 0
	keep_running = 1
	while keep_running==1:
		#(self, image, parent, color, depth, rel_loc, px, py, sx, sy)
		root = QuadNode(im, None, im_color, 0, "", im.size[0]/2, im.size[1]/2, im.size[0]/2, im.size[1]/2)
		root.build()
		root.traverse(None,None,None,None) 
		#safe_point_test is called within traverse
		#if there's a pixel removal, safe_point_test will set keep_running = 1
		#since it is not complete, an iteration limit is used
		iteration = iteration + 1
		if(iteration>max_iteration):
			keep_running = 0
	pp.imshow(im, origin="lower")
	
	#for testing, this will build the quadtree, echo the node and show an image created using the quadtree
	root = QuadNode(im, None, im_color, 0, "", im.size[0]/2, im.size[1]/2, im.size[0]/2, im.size[1]/2)
	root.build()
	#root.echo()
	root.draw(im2)
	pp.figure()
	pp.imshow(im2, origin="lower")


#create the node class
class QuadNode:
	"""
	ebc, sbc, wbc, nbc #boundary colors, color of 2 quads on each side
	color #this node color
	nw, ne, ws, se #QuadNode subsections
	px, py #the center position
	sx, sy #the size of the section
	depth #determine the scale
	"""
	
	def __init__(self, image, parent, color, depth, rel_loc, px, py, sx, sy):
		self.image = image
		self.parent = parent
		self.color = color
		self.depth = depth
		self.relative_location = rel_loc
		self.px = px
		self.py = py
		self.sx = sx
		self.sy = sy
		self.nw = None
		self.ne = None
		self.sw = None
		self.se = None
		self.ebc = None
		self.sbc = None
		self.wbc = None
		self.nbc = None
	
	#have method that extracts a section based on level from pixels
	#variable: level/depth (1,2,3,etc) for how big the section is
	#use histogram for that section, check only for two value, 0 and 255
	#if only 0 has value -> black node
	#if only 255 has value -> white node
	#if both have value -> gray node, continue to subdivide into new sections with smaller level
	#store into the quadnode class
	def build(self):
		#check children
		c_depth = self.depth + 1
		c_sx = (self.sx/2)
		c_sy = (self.sy/2)
		#divide into subsection
		if(self.color==2 and c_depth<max_depth and self.sx>=1 and self.sy>=1): #gray, within max_depth, and at least 1 pixel	
			#(px-sx, py-sy) to (px, py) nw
			nw_px = self.px-c_sx
			nw_py = self.py-c_sy
			#the image data
			nw_im = self.image.crop((self.px-self.sx, self.py-self.sy, self.px, self.py))
			nw_hist = nw_im.histogram()
			#default white
			nw_color = 1
			if(nw_hist[0]>0 and nw_hist[255]>0):
				nw_color = 2 #gray
			elif(nw_hist[0]>0):
				nw_color = 0 #black
			self.nw = QuadNode(self.image, self, nw_color, c_depth, "nw", nw_px, nw_py, c_sx, c_sy)
			if nw_color == 2:
				self.nw.build()
			#end of nw
			
			#(px, py-sy) to (px+sx,py) ne
			ne_px = self.px+c_sx
			ne_py = self.py-c_sy
			#the image data
			ne_im = self.image.crop((self.px, self.py-self.sy, self.px+self.sx, self.py))
			ne_hist = ne_im.histogram()
			#default white
			ne_color = 1
			if(ne_hist[0]>0 and ne_hist[255]>0):
				ne_color = 2 #gray
			elif(ne_hist[0]>0):
				ne_color = 0 #black
			self.ne = QuadNode(self.image, self, ne_color, c_depth, "ne", ne_px, ne_py, c_sx, c_sy)
			if ne_color == 2:
				self.ne.build()
			#end of ne
			
			#(px-sx, py) to (px, py+sy) sw
			sw_px = self.px-c_sx
			sw_py = self.py+c_sy
			#the image data
			sw_im = self.image.crop((self.px-self.sx, self.py, self.px, self.py+self.sy))
			sw_hist = sw_im.histogram()
			#default white
			sw_color = 1
			if(sw_hist[0]>0 and sw_hist[255]>0):
				sw_color = 2 #gray
			elif(sw_hist[0]>0):
				sw_color = 0 #black
			self.sw = QuadNode(self.image, self, sw_color, c_depth, "sw", sw_px, sw_py, c_sx, c_sy)
			if sw_color == 2:
				self.sw.build()
			#end of sw
			
			#(px, py) to (px+sx, py+sy) se
			se_px = self.px+c_sx
			se_py = self.py+c_sy
			#the image data
			se_im = self.image.crop((self.px, self.py, self.px+self.sx, self.py+self.sy))
			se_hist = se_im.histogram()
			#default white
			se_color = 1
			if(se_hist[0]>0 and se_hist[255]>0):
				se_color = 2 #gray
			elif(se_hist[0]>0):
				se_color = 0 #black
			self.se = QuadNode(self.image, self, se_color, c_depth, "se", se_px, se_py, c_sx, c_sy)
			if se_color == 2: #if gray continue building
				self.se.build()
			#end of se
			
			#set the boundary colors to gray
			self.ebc = 2
			self.sbc = 2
			self.wbc = 2
			self.nbc = 2
			
			#reset based on the children
			if(se_color!=2 and ne_color==se_color): 
				self.ebc = se_color
			elif(self.ne.sbc==self.se.ebc):
				self.ebc = self.se.ebc
			
			if(sw_color!=2 and sw_color==se_color): 
				self.sbc = sw_color
			elif(self.sw.sbc==self.se.sbc):
				self.sbc = self.se.sbc
			
			if(nw_color!=2 and nw_color==sw_color): 
				self.wbc = nw_color
			elif(self.nw.wbc==self.sw.wbc):
				self.wbc = self.sw.sbc
				
			if(ne_color!=2 and nw_color==ne_color): 
				self.nbc = nw_color
			elif(self.nw.nbc==self.ne.nbc):
				self.nbc = self.ne.nbc
			
		else:
			#other wise set boundary colors to its own color
			self.ebc = self.color
			self.sbc = self.color
			self.wbc = self.color
			self.nbc = self.color
		
	def traverse(self, w, n, e, s):
		if(self.nw!=None and self.ne!=None and self.sw!=None and self.se!=None):
			#nw
			if(self.nw.color==2) or (self.nw.color==0 and (w==None or w.ebc!=0 or n==None or n.sbc!=0 or self.ne.wbc!=0 or self.sw.nbc!=0)):
				self.nw.traverse(w, n, self.ne, self.sw)
			#ne
			if(self.ne.color==2) or (self.ne.color==0 and (self.nw.ebc!=0 or n==None or n.sbc!=0 or e==None or e.wbc!=0 or self.se.nbc!=0)):
				self.ne.traverse(self.nw, n, e, self.se)
			#sw
			if(self.sw.color==2) or (self.sw.color==0 and (w==None or w.ebc!=0 or self.nw.sbc!=0 or self.se.wbc!=0 or s==None or s.nbc!=0)):
				self.sw.traverse(w, self.nw, self.se, s)
			#se
			if(self.se.color==2) or (self.nw.color==0 and (self.sw.ebc!=0 or self.ne.sbc!=0 or e==None or e.wbc!=0 or s==none or s.nbc!=0)):
				self.se.traverse(self.sw, self.ne, e, s)
		else:
			self.safe_point_test()
		
	def safe_point_test(self):
		#this is pixel based operation
		#for every pixel in this node
		#get pixel by position
		#if pixel is black, calculate safe_point_index
		#   if safe_point_index==1, set pixel color to white
		#   if safe_point_index==2, iterate cases for further checking
		global keep_running
		lx = self.parent.px
		ly = self.parent.py
		lsx = self.parent.sx
		lsy = self.parent.sy
		if(self.parent.nw == self): 
			lx = lx-lsx
			ly = ly-lsy
		if(self.parent.ne == self): 
			lx = lx
			ly = ly-lsy
		if(self.parent.sw == self): 
			lx = lx-lsx
			ly = ly
		if(self.parent.se == self): 
			lx = lx
			ly = ly
		#top
		for j in range(lsx):
			for k in range(lsy):
				pi = self.point_index(lx+j, ly+k)
				#print self.relative_location+" "+color(self.color)+" "+str(self.image.getpixel((lx+j, ly+k)))+" "+str(pi)
				if(safe_point_lookup[pi]==1):
					self.image.putpixel((lx+j, ly+k), 255)
				elif(safe_point_lookup[pi]==2):
					self.image.putpixel((lx+j, ly+k), 255)
					#should iterate the 5x5 pixel check here, using the (lx+j, ly+k) as the relative position
					#print("safe_point_test(" + color(self.color) + "|" + str(self.px) + "," + str(self.py) + "|" + str(self.depth) + "|" + self.relative_location + "|"+ str(self.point_color(self.px, self.py)) + ")" )
		
	def echo(self):
		space = " "
		for j in range(self.depth):
			space += " "
		print(space + "(" + color(self.color) + "|" + str(self.px) + "," + str(self.py) + "|" + str(self.depth) + "|" + self.relative_location + ")")
		#then print children
		if(self.nw!=None):	self.nw.echo()
		if(self.ne!=None):	self.ne.echo()
		if(self.sw!=None):	self.sw.echo()
		if(self.se!=None):	self.se.echo()
	def draw(self, buffer):
		if(self.nw==None and self.ne==None and self.sw==None and self.se==None):
			#draw
			if(self.color==0):
				#pencil = id.Draw(buffer)
				#draw from parent
				lx = self.parent.px
				ly = self.parent.py
				lsx = self.parent.sx
				lsy = self.parent.sy
				if(self.parent.nw == self): 
					lx = lx-lsx
					ly = ly-lsy
				if(self.parent.ne == self): 
					lx = lx
					ly = ly-lsy
				if(self.parent.sw == self): 
					lx = lx-lsx
					ly = ly
				if(self.parent.se == self): 
					lx = lx
					ly = ly
				for j in range(lsx):
					for k in range(lsy):
						buffer.putpixel((lx+j, ly+k), 0)
		else:
			if(self.nw!=None):	self.nw.draw(buffer)
			if(self.ne!=None):	self.ne.draw(buffer)
			if(self.sw!=None):	self.sw.draw(buffer)
			if(self.se!=None):	self.se.draw(buffer)
	def point_color(self, vx, vy):
		return self.image.getpixel((vx, vy))
	def point_index(self, vx, vy):
		total = 0
		if(vx>0 and vy>0):
			if(self.image.getpixel((vx-1,vy-1))==0): total += 1
		if(vx>0):
			if(self.image.getpixel((vx-1,vy))==0): total += 8
			if(vy+1<self.image.size[1] and (self.image.getpixel((vx-1, vy+1))==0)): total += 32
		if(vy>0):
			if(self.image.getpixel((vx, vy-1))==0): total += 2
			if(vx+1<self.image.size[0] and (self.image.getpixel((vx+1, vy-1))==0)): total += 4
		if(vx+1<self.image.size[0] and vy+1<self.image.size[1] and (self.image.getpixel((vx+1, vy+1))==0)): total += 128
		if(vx+1<self.image.size[0] and (self.image.getpixel((vx+1, vy))==0)): total += 16
		if(vy+1<self.image.size[1] and (self.image.getpixel((vx, vy+1))==0)): total += 64
		return total
		
def color(val):
	if(val==0): return "black"
	elif(val==1): return "white"
	else: return "gray"
		
if __name__ == "__main__":
    #quad(int(sys.argv[1]))
	quad()