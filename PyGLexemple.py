#Import OpenGL and GLU.  Don't import GLUT because it is ancient, broken, inflexible, and poorly
#designed--and we aren't using it.
from OpenGL.GL import *
from OpenGL.GLU import *
#Import PyGame.  We'll mostly just use this to make a window.  Also import all the local
#declarations (e.g. pygame.KEYDOWN, etc.), so that we don't have to keep typing "pygame." in front
#of everything.  E.g., now we can do "KEYDOWN" instead of "pygame.KEYDOWN".
import pygame
from pygame.locals import *
#Import some other useful modules
import sys, os, traceback
#Center the window on the screen, if we're on Windows, which supports it.
if sys.platform in ["win32","win64"]: os.environ["SDL_VIDEO_CENTERED"]="1"
#Import sin, cos, radians, degrees, etc.
from math import *
#Initialize PyGame.  You could also call "pygame.init()", but in my experience this can be faster
#(since you aren't initializing *everything*) and more portable (since some modules may require
#extra dependencies).
pygame.display.init()
pygame.font.init()

#Screen configuration
screen_size = [800,600]
multisample = 0
#Set the window's icon, as applicable, to be just a transparent square.
icon = pygame.Surface((1,1)); icon.set_alpha(0); pygame.display.set_icon(icon)
#Set the title of the window.
pygame.display.set_caption("PyOpenGL Example - Ian Mallett - v.1.0.0 - 2013")
#Set the window to be multisampled.  This does depth testing at a higher resolution, leading to
#smooth, antialiased edges.  Most computers support at least multisample=4, and most support more
#(e.g. mine does 16).
if multisample:
    pygame.display.gl_set_attribute(GL_MULTISAMPLEBUFFERS,1)
    pygame.display.gl_set_attribute(GL_MULTISAMPLESAMPLES,multisample)
#Create the window of the requested size.  The pygame.OPENGL flag tells it to allow OpenGL to write
#directly to the window context.  The pygame.DOUBLEBUF flag tells it to make the window
#doublebuffered.  This causes the screen to only show a completed image.  This function actually
#returns a "surface" object, but it isn't useful for OpenGL programs.
pygame.display.set_mode(screen_size,OPENGL|DOUBLEBUF)

#If we draw a new pixel, we want to blend the new pixel with whatever is already there.  This allows
#for transparency, among other things.  Since everything here is fully opaque, we don't actually
#*need* this right now.
##glEnable(GL_BLEND)
##glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)

#Enable textured objects.  The glTexEnvi calls set up texturing in an intuitive way.  Again, since
#nothing here is textured, we don't actually *need* this right now.
##glEnable(GL_TEXTURE_2D)
##glTexEnvi(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_MODULATE)
##glTexEnvi(GL_POINT_SPRITE,GL_COORD_REPLACE,GL_TRUE)

#This requests that OpenGL make interpolation (filling in triangles) happen in the nicest way
#possible.  It's not guaranteed to happen; it's a request.
glHint(GL_PERSPECTIVE_CORRECTION_HINT,GL_NICEST)
#This enables depth testing (so that closer objects are always drawn in front of farther objects).
#If depth testing is not enabled, then objects are drawn "over" each other in the order you draw
#them.  For most 3D rendering, you'll want depth testing enabled.
glEnable(GL_DEPTH_TEST)

#This concludes setup; the program itself will be a single triangle drawn in white at positions
#(0.0,0.0,0.0), (0.8,0.0,0.0), (0.0,0.0,0.4), along with a red, green, and blue line segments
#showing the axes.

#I find that an intuitive basic setup for the camera (where you're looking from) is to have the
#viewer located on the surface of a sphere surrounding everything.  You can change your position on
#the sphere, and thus fly around the scene.  To do this, I put the camera in (a kind of) spherical
#coordinates.

camera_rot = [30.0,20.0]      #The spherical coordinates' angles (degrees).
camera_radius = 3.0           #The sphere's radius
camera_center = [0.0,0.0,0.0] #The sphere's center
def get_input():
    global camera_rot, camera_radius
    #Input in PyGame is pretty straightforward.  For now, we are concerned only with key and mouse
    #input.  Whenever anything *happens* (move the mouse, click, etc.), an "event" happens.  You get
    #a list of the events that happened by calling "pygame.event.get()".  You can also query the
    #*state* of anything by checking it specifically.

    #Check the *state* of the keys, the mouse buttons, and the mouse's position within the window.
    keys_pressed = pygame.key.get_pressed()
    mouse_buttons = pygame.mouse.get_pressed()
    mouse_position = pygame.mouse.get_pos()
    #Check how much the mouse moved since you last called this function.
    mouse_rel = pygame.mouse.get_rel()
    #List all the events that happened.
    for event in pygame.event.get():
        #Clicked the little "X"; close the window (return False breaks the main loop).
        if   event.type == QUIT: return False
        #If the user pressed a key:
        elif event.type == KEYDOWN:
            #If the user pressed the escape key, close the window.
            if   event.key == K_ESCAPE: return False
        #If the user "clicked" the scroll wheel forward or backward:
        elif event.type == MOUSEBUTTONDOWN:
            #Zoom in
            if   event.button == 4: camera_radius *= 0.9
            #Or out.
            elif event.button == 5: camera_radius /= 0.9
    #If the user is left-clicking, then move the camera about in the spherical coordinates.
    if mouse_buttons[0]:
        camera_rot[0] += mouse_rel[0]
        camera_rot[1] += mouse_rel[1]
    return True
def draw():
    #Clear the screen's color and depth buffers so we have a fresh space to draw geometry onto.
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    #Setup the viewport (the area of the window to draw into)
    glViewport(0,0,screen_size[0],screen_size[1])
    #Change the matrix mode to the projection matrix (all subsequent calls that change matrices will
    #change the projection matrix).  The projection matrix should be made responsible for taking all
    #the geometry in the 3D world and then distorting it so that it is in perspective on the screen.
    glMatrixMode(GL_PROJECTION)
    #Set the current matrix (the projection matrix) to be the identity matrix.
    glLoadIdentity()
    #Multiply the current matrix (the projection matrix) by a matrix that projects everything like a
    #camera would.  Basically, this makes everything look like it's in perspective.  In this case,
    #the camera has a (vertical) field of view of 45 degrees, an aspect ratio of 800.0/600.0, a near
    #clipping plane of 0.1, and a far clipping plane of 100.0.  The clipping planes tell you how
    #close and far away from the camera you can see things.  Ideally, you'd set them to 0.0 and
    #infinity, but the clipping planes also affect the depth buffer; setting them farther apart
    #means objects don't occlude each other as correctly (the depth buffer is stretched over a
    #larger distance).  The general rule is to set the near clipping plane as large as possible (and
    #*never* to 0.0), and then make your far plane reasonably small.
    gluPerspective(45, float(screen_size[0])/float(screen_size[1]), 0.1,100.0)
    #Change the matrix mode to the modelview matrix (all subsequent calls that change matrices will
    #change the modelview matrix).  The modelview matrix should be made responsible for moving
    #things around the world (the "model" part of the name) and also making it look like the camera
    #is in a particular position (the "view" part of the name).
    glMatrixMode(GL_MODELVIEW)
    #Set the current matrix (the modelview matrix) to be the identity matrix.
    glLoadIdentity()

    #The matrices stay the way they are until they are changed.  Since the projection matrix doesn't
    #actually change from frame to frame, one *could* only set it once.  You will see this approach
    #in other tutorials.  This isn't a good idea, since more advanced techniques (e.g. image-space
    #techniques) require the projection matrix to constantly change.

    #Set the camera's position to be in spherical coordinates.  These aren't typical spherical
    #coordinates, since I take the elevation angle (camera_rot[1]) to be 0.0 at the horizon.  I find
    #this more intuitive, but you can easily change it to your favorite parameterization by
    #exchanging sines and cosines.
    camera_pos = [
        camera_center[0] + camera_radius*cos(radians(camera_rot[0]))*cos(radians(camera_rot[1])),
        camera_center[1] + camera_radius                            *sin(radians(camera_rot[1])),
        camera_center[2] + camera_radius*sin(radians(camera_rot[0]))*cos(radians(camera_rot[1]))
    ]
    #This multiplies the current matrix (the modelview matrix) by a matrix that makes it *look like*
    #all subsequent draw calls had the camera at the given position and direction.  In reality, it
    #actually rotates and translates *the whole world* so that it *looks* that way, but the effect
    #is the same.  Here, the camera has position "camera_pos" and is oriented so that it is looking
    #towards position "camera_center".  The last three arguments tell it which way is up.
    gluLookAt(
        camera_pos[0],camera_pos[1],camera_pos[2],
        camera_center[0],camera_center[1],camera_center[2],
        0,1,0
    )

    #Okay!  Let's start *actually drawing stuff*!  We use "immediate mode" OpenGL here, which is
    #obsoleted by vertex arrays and VBOs.  Still, immediate mode is far more intuitive, so it is the
    #method we'll use here.

    #Set the color to white.  All subsequent geometry we draw will be white.  This is actually the
    #default, so we didn't *need* to do this.
    glColor3f(1,1,1)

    #Start drawing triangles.  Each subsequent triplet of glVertex*() calls will draw one triangle.
    glBegin(GL_TRIANGLES)
    glVertex3f(0.0,0.0,0.0) #Make a vertex at (0.0,0.0,0.0)
    glVertex3f(0.8,0.0,0.0) #Make a vertex at (0.8,0.0,0.0)
    glVertex3f(0.0,0.0,0.4) #Make a vertex at (0.0,0.0,0.4)
    #Now that we've made one triplet of glVertex*() calls, it will draw one (white) triangle between
    #those three points.  We're done drawing triangles; tell OpenGL so.
    glEnd()

    #Start drawing lines.  Each subsequent pair of glVertex*() calls will draw one line.
    glBegin(GL_LINES)
    #Change the color to red.  All subsequent geometry we draw will be red.
    glColor3f(1,0,0)
    #Make two vertices, thereby drawing a (red) line.
    glVertex(0,0,0); glVertex3f(1,0,0)
    #Change the color to green.  All subsequent geometry we draw will be green.
    glColor3f(0,1,0)
    #Make two vertices, thereby drawing a (green) line.
    glVertex(0,0,0); glVertex3f(0,1,0)
    #Change the color to blue.  All subsequent geometry we draw will be blue.
    glColor3f(0,0,1)
    #Make two vertices, thereby drawing a (blue) line.
    glVertex(0,0,0); glVertex3f(0,0,1)
    #Change the color to white again.  All subsequent geometry we draw will be white.  Strictly
    #speaking this isn't required (since we reset the color on line 166 before we draw anything
    #again).  However, it is good practice to reset the color to white, since forgetting to can be a
    #hard-to-track-down bug (e.g. when combining with texturing).
    glColor3f(1,1,1)
    #We're done drawing lines; tell OpenGL so.
    glEnd()

    #Flip the buffer (draw the internal memory we've been using onto the screen).  This is why we
    #passed pygame.DOUBLEBUF when we created the window.
    pygame.display.flip()
def main():
    clock = pygame.time.Clock()
    while True:
        if not get_input(): break
        draw()
        clock.tick(120) #Regulate the framerate to be as close as possible to 60Hz.
    pygame.quit()
if __name__ == "__main__":
    try:
        main()
    except:
        traceback.print_exc()
        pygame.quit()
        input()
