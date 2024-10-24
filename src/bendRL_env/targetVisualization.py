import sys  # for the popup display
import random  # to generate random numbers
import tkinter
from tkinter import *
import numpy as np
import time

# TO DO
# - Figure how to match the dot in the screen to align with goal coordinates


class TargetDisplay:
    """ This is a full size display of a red dot on a background,
    in a popup window."""

    def __init__(self, window, bk_colour="white", circle_colour="red", env_type="static", pos=None):
        """ The constructor of the RedDotDisplay. The size of the dot and
        the background color are set to 50 px and black, for now, but this
        can be changed by accessing the parameters. The window is set to
        fit the full screen it will be displayed in.
        window: a tkinter object, has to be created then passed as a parameter
        dot_pos: Center of dot, given in TCP coordinates"""

        self.root = window
        self.display_width = self.root.winfo_screenwidth()
        self.display_height = self.root.winfo_screenheight()
        self.display_size = str(self.display_width) + "x" + str(self.display_height)
        # self.root.geometry(self.display_size)

        w0, h0 = 2560, 1920
        w1, h1 = 1440, 1200
        self.root.geometry(f"{w1}x{h1}+{w0}+0") # <- this is the key, offset to the right by w0
        self.root.attributes("-fullscreen", True)

        self.root.title("Target")
        self.root.configure(bg=bk_colour)

        self.env_type = env_type # Should be "reaching", "tracking" or "static"
        self.dot_radius = 100

        self.velocity_x = 0
        self.velocity_y = 0
        self.x_coord = 1900 / 2
        self.y_coord = 1100 / 2
        # the following won't work for some reason
        # self.x_coord = self.display_width
        # self.y_coord = self.display_height
        if pos is None:
            # sets the default position to the middle
            self.dot_pos = [self.x_coord, self.y_coord]
        else:
            # sets the default position to the position provided by user
            print("pos set " )
            print(pos)
            self.dot_pos = pos[0], pos[1]

        # Creating the red dot
        self.canvas = Canvas(self.root, width=1900, height=1100, background=bk_colour)
        self.canvas.pack()

        print(self.dot_pos)

        x0 = self.dot_pos[0] - self.dot_radius
        y0 = self.dot_pos[1] - self.dot_radius
        x1 = self.dot_pos[0] + self.dot_radius
        y1 = self.dot_pos[1] + self.dot_radius

        self.red_dot = self.canvas.create_oval(x0, y0, x1, y1, fill=circle_colour, outline=circle_colour)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def reset(self, env_type="static"):
        """this should be called at the end of a training episode
        to reset the position back to the center of the screen"""
        # The Reacher environment needs to move the target at the start of an episode
        self.velocity_x = 0
        self.velocity_y = 0
        if env_type == "reaching":
            self.move(env_type="reaching")
        # Both tracking and static environment want to reset back to the default position
        else:
            coordinates = self.canvas.coords(self.red_dot)
            offset_x = self.dot_pos[0] - (coordinates[0] + self.dot_radius)
            offset_y = self.dot_pos[1] - (coordinates[1] + self.dot_radius)

            self.canvas.move(self.red_dot, offset_x, offset_y)
            self.canvas.update()
            self.canvas.update_idletasks()
            self.root.update()
            self.root.update_idletasks()

    def _compute(self, env_type="reaching", random_pos=False):
        """ In a reaching environment, this moves the target to a random position (random_pos=True) but setting
        random_pos = False puts it into the position provided in the constructor (or default).
        In the tracking environment, this sets the new random direction of the target.
        The actual move operation is performed by regenerate, which is called inside the environments
        _compute is never called in a 'static' env_type."""

        coordinates = self.canvas.coords(self.red_dot)
        # these are hardcoded because need a buffer on the size
        width = 1900
        height = 1100
        if env_type == "reaching":
            if not random_pos:
            # sets the default position
                self.offset_x = self.dot_pos[0] - (coordinates[0] + self.dot_radius)
                self.offset_y = self.dot_pos[1] - (coordinates[1] + self.dot_radius)
            # sets a new random position
            else:
                # here we will generate random positions for the dot
                x_coord = coordinates[0] + self.dot_radius
                y_coord = coordinates[1] + self.dot_radius

                self.offset_x = random.randint(-x_coord + self.dot_radius, (width - x_coord) - self.dot_radius)
                self.offset_y = random.randint(-y_coord + self.dot_radius, (height - y_coord) - self.dot_radius)

            # print("offset x: " + str(self.offset_x) + ", offset y: " + str(self.offset_y))

        # if we are in tracking mode, we want the dot to move a small amount constantly...
        elif env_type == "tracking":
            # this velocity is in pixels, and pixels have to be integers
            self.velocity_x = random.randint(-3, 3) * 4.5 #-3 to 3 covers almost to 180 degrees (rad = 3.14 for 180)
            self.velocity_y = random.randint(-3, 3) * 4.5
            # Next 3 lines from:
            # https://github.com/YufengYuan/ur5_async_rl/blob/main/envs/visual_ur5_reacher/monitor_communicator.py#L56
            # velocity = np.sqrt(self.velocity_x ** 2 + self.velocity_y ** 2)
            # self.velocity_x /= velocity
            # self.velocity_y /= velocity
        else:
            print("Wrong environment type specified. Possible values are 'reaching' or 'tracking'")

    def move(self, env_type="static"):
        """This performs the move to the new position, computed in _compute()"""
        width = 1900
        height = 1100
        # get actual coordinates of center of dot
        coordinates = self.canvas.coords(self.red_dot)
        x_coord = coordinates[0] + self.dot_radius
        y_coord = coordinates[1] + self.dot_radius

        if env_type == "reaching":
            self._compute(env_type="reaching", random_pos=True)
            self.canvas.move(self.red_dot, self.offset_x, self.offset_y)
        elif env_type == "tracking":
            if self.velocity_x == 0 or self.velocity_y == 0:
                self._compute(env_type="tracking", random_pos=True)
            # Check if we're about to go out of bounds in x direction
            if (x_coord + self.velocity_x) > (width - self.dot_radius):
                self.velocity_x = -self.velocity_x
            elif (x_coord + self.velocity_x) < (0 + self.dot_radius):
                self.velocity_x = -self.velocity_x
            # Check if we're about to go out of bounds in y direction
            if (y_coord + self.velocity_y) > (height - self.dot_radius):
                self.velocity_y = -self.velocity_y
            elif (y_coord + self.velocity_y) < (0 + self.dot_radius):
                self.velocity_y = -self.velocity_y
            # Otherwise, NO need to call reset, just keep moving until we hit a wall
            self.canvas.move(self.red_dot, self.velocity_x, self.velocity_y)
        elif env_type == "static":
            pass

        self.canvas.update()
        self.canvas.update_idletasks()
        self.root.update()
        self.root.update_idletasks()

    def on_closing(self):
        self.root.destroy()


if __name__ == '__main__':
    root = tkinter.Tk()
    red_dot = TargetDisplay(root)

    # red_dot.regenerate(random_pos=False)
    # root.protocol("WM_DELETE_WINDOW", red_dot.on_closing)
    # root.mainloop()
    #
    # time.sleep(2)
    # red_dot.regenerate()


