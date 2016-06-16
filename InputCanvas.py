# -*- coding:utf-8 -*-
from PIL import Image
import tkinter
import io


class InputCanvas(tkinter.Canvas):
    def on_pressed(self, event):
        self.sx, self.sy = event.x, event.y
        w = self.linewidth / 2
        self.create_oval(self.sx-w, self.sy-w, self.sx+w, self.sy+w,
                                fill = self.linecolor,
                                width = 0)

    def on_dragged(self, event):
        ex, ey = event.x, event.y
        self.create_line(self.sx, self.sy, ex, ey,
                                fill = self.linecolor,
                                width = self.linewidth)
        w = self.linewidth / 2
        self.create_oval(ex-w, ey-w, ex+w, ey+w,
                                fill = self.linecolor,
                                width = 0)
        self.sx, self.sy = event.x, event.y

    def getImage(self):
        ps = self.postscript(colormode='color')
        return Image.open(io.BytesIO(ps.encode('utf-8')))

    def clear(self):
        self.delete('all')

    def __init__(self, window, width, height, bg = "white"):
        super(InputCanvas, self).__init__(window, bg = bg, width = width, height = height)

        self.bind("<ButtonPress-1>", self.on_pressed)
        self.bind("<B1-Motion>", self.on_dragged)

        self.linecolor = "black"
        self.linewidth = 22
