from tkinter import *
 
class BasicMishaCanvas(Canvas):
 
    def __init__(self, master, rows, cols, cellsize = 10):
        self.rows = rows
        self.cols = cols
        self.cellsize = cellsize
        self.width = self.cellsize * cols
        self.height = self.cellsize * rows
        Canvas.__init__(self, master, width = self.width, height = self.height,
                        borderwidth=0, background='white')
        self.pack()
        #  Create the 2D array of rectangles--white to start with
        self.rects = self.makeRectangles()
 
    def makeRectangles(self):
        returnme = [[0 for x in range(self.cols)] for y in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                xup = r  * self.cellsize
                yleft = c  * self.cellsize
                returnme[r][c] = self.create_rectangle(yleft, xup, yleft + self.cellsize,
                         xup + self.cellsize,
                         fill = "white")
        return returnme
     
    def fillPoint(self, x, y):
        self.itemconfig(self.rects[x][y], fill = "black")
 
    def erasePoint(self, x, y):
        self.itemconfig(self.rects[x][y], fill = "white")
 
    def isFilled(self, x, y):
        if self.itemcget(self.rects[x][y],"fill") == "black":
            return True
        return False
 
    def isValid(self, x, y):
        if 0 <= x and x < self.rows and 0 <= y and y < self.cols:
            return True
        return False
 
    def cell_coords(self, mousex, mousey):
        # Reversed, to change from canvas coordinates to 2D array coordinates
        return (mousey//self.cellsize, mousex//self.cellsize)
