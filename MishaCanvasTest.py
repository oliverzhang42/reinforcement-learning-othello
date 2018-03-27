from MishaCanvas import *
from tkinter import *
 
root = Tk()
# Grid is 10x15, cell size 20 pixels, white to start with
#  It will pack itself into the root
mc = BasicMishaCanvas(root, 28, 28, 10)
 
# Bind a function to a click to the canvas
def canvasclick(event):
    x, y = mc.cell_coords(event.x, event.y)
    mc.fillPoint(x, y)

def canvasdrag(event):
    x, y = mc.cell_coords(event.x, event.y)
    mc.fillPoint(x, y)

def convertToImage(event):
    for i in range(28):
        array = []
        for j in range(28):
            array.append(int(mc.isFilled(i,j)))
        #print(array)
        img.append(array)
    
mc.bind("<Button-1>", canvasclick)
mc.bind("<B1-Motion>", canvasdrag) 
mc.bind("<Double-Button-1>", convertToImage)

img = []


