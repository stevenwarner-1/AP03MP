""" This is a demonstration of some basic data handling of Meteosat images
    which might be useful as a basis for the AP03 miniproject 
 
    There are two class definitions, Image and Geo, followed by the main program
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys

#-------------------------------------------------------------------------------
class Image:  
  """ Image data and methods 

  DATA
    tem   boo : True=brightness temperature image, False=radiance image
    nx    int : No of horizontal pixels in images
    ny    int : No of vertical pixels in images
    ixoff int : pixel offset of left edge of box
    iyoff int : pixel offset of bottom edge of box
    data  flt : np.(ny,nx) image data as 2D array
    title str : image title 

  METHODS 
    __init__ : Initialise new Img object
        disp : Display full disk image
        clip : Create new image from subset of original image
      bright : Convert radiance image to Brightness Temperature image
      cloud  : Remove cloud from image

  HISTORY
      v16Oct20 : AD Original version
  """

  def __init__(self,imgfil,title=None):
    """ Initialise new image object 

    PARAMETERS
      imgfil str : file name for image
      title  str : description of image

    DESCRIPTION
      Read in image data as 2D array of floating values.
      If called with imgfil==None this just creates a new 'empty' object 
      If called without title parameter, the title will be taken from the file
    """

    self.tem = False    # Assume any new image is radiance
    self.ixoff = 0
    self.iyoff = 0
    if imgfil is None:  # Create an empty object
      self.nx = 0
      self.ny = 0
      self.data = 0
      self.title = ''
    else:               # Read in image data from file
      with open(imgfil) as f:
        imgtitle = f.readline()
        if title is None: self.title=imgtitle
        else: self.title=title
        nx, ny = np.fromfile(f,dtype=int,count=2,sep=" ")  
        self.nx = nx
        self.ny = ny
        imgdata = np.fromfile(f, dtype=float, count=nx*ny, sep=" ")
        self.data = imgdata.reshape((ny,nx))

  def disp(self,window=1,box=None):
    """ Display the image 

    PARAMETERS
      window int : plot window for display
      box        : box coordinates, dictionary containing keys
        'xmin'  int : Pixel# of left edge of box
        'xmax'  int : Pixel# of right edge of box
        'ymin'  int : Pixel# of bottom edge of box
        'ymax'  int : Pixel# of top edge of box
        'color' str : (optional, default='white') color of box

    DESCRIPTION
      Basic function is to display the stored image self.data
      Can also superimpose a rectangular box defined by keys 'xm

    """
    plt.figure(window)
    plt.clf()
    plt.axis('off')                  # no axes required
    plt.title(self.title)
    if self.tem:                     # colour scale for bright.tem image
      colours = [(1,1,1),(1,0,1),(0,0,1),(0,1,1),(0,1,0),(1,1,0),
                 (1,0,0),(0,0,0)]
      tnorm = plt.Normalize(230,320) # range 230-320K
      tcmap = LinearSegmentedColormap.from_list('tem_colours', colours)
      plt.imshow(self.data, origin='lower', cmap=tcmap, norm=tnorm)
      plt.colorbar(label='Temperature [K]')
    else:                            # use gray scale for radiance image
      plt.imshow(self.data, origin='lower', cmap='gray')
      plt.colorbar(label='Radiance')
    if box is not None:
      xmin = box['xmin'] - self.ixoff
      xmax = box['xmax'] - self.ixoff
      ymin = box['ymin'] - self.iyoff
      ymax = box['ymax'] - self.iyoff
      if 'color' in box: col=box['color']
      else: col='white'
      plt.plot([xmin,xmax,xmax,xmin,xmin],[ymin,ymin,ymax,ymax,ymin],color=col)
           
    plt.tight_layout(pad=0.05)       # maximise size of image within window
    plt.show()

  def clip ( self, box, title=None ):
    """ Create a new image from a subset of another image

    PARAMETERS
      box   dict : Subsect coordinates, as defined in self.disp
      title str  : Title for new image (else copy original image title)

    RETURNS
      newimg : Image object

    DESCRIPTION
      Creates a new image object from a rectangular subset of an existing image
      Note that pixel numbers from the original image have to be preserved, via
      self.ixoff and self.iyoff, in order for the geolocation to work on the
      subset image
    """
    xmin = box['xmin']
    xmax = box['xmax']
    ymin = box['ymin']
    ymax = box['ymax']
    newimg = Image(None)  
    newimg.nx = xmax-xmin 
    newimg.ny = ymax-ymin
    newimg.ixoff = xmin
    newimg.iyoff = ymin
    y1 = ymin - self.iyoff
    y2 = ymax - self.iyoff
    x1 = xmin - self.ixoff
    x2 = xmax - self.ixoff
    newimg.data = self.data[y1:y2,x1:x2]
    if title is None: newimg.title = self.title
    else: newimg.title = title
    return newimg

  def bright ( self, wavelength ):
    """ Convert radiance image to Brightness Temperature image

    PARAMETERS
      wavelength flt : characteristic wavelength [microns] for channel

    DESCRIPTION
      Convert image from radiance [W/(m2 sr um) to brightness temperature [K]
      using the inverse of the Planck function at the characteristic 
      wavelength. 
    """
    # Local constants
    H = 6.63e-34       # Planck constant       [m2 kg / s]
    C = 3.00e8         # Speed of light        [m / s]
    K = 1.38e-23       # Boltzmann constant    [m2 kg /s2 /K]
    R1 = H * C / K     # Intermediate Constant [m K]
    R2 = 2 * H * C**2  # Intermediate Constant [m4 kg / s3]
    if self.tem:
      print('Apparently already a brightness temperature image')
    else: 
      w = wavelength * 1.0e-6  # convert microns to metres
      self.data = R1 / w / np.log( 1.0 + R2/(w**5 * self.data*1e6) )
      self.tem = True
      
      
  def cloud ( self, method, time ):
    """ Remove cloud from given image

    PARAMETERS
      method: type of cloud removal: day (1) or night (0)
      time: the Z time (in hours) that the image was taken at
      temp_threshold: ch9 brightness temperature below which cloud is designated (example value = 280K for daytime)
      rad_threshold: ch1/ch9 threshold, above/below which cloud is designated (example value = 100 for daytime)
      

    DESCRIPTION
      Uses a cloud filter based on radiance in channels 1 (day) and 4 (night) and brightness temperature in
      channel 9 to identify and remove cloud from the image
    """
    
    image = self
    # determining time of image to be studied
    hour = time # using the 24 hour structure
    hour_str = str(hour)

    if len(str(hour)) == 1:
      hour_str = '0{}'.format(hour)

    #input('1) Load Ch9 infrared image... ')
    c9b = Image('msg_c09_z{}.img'.format(hour_str))  # Load Ch9 image
    #c9b.disp(window=3)   # display in a different window

    #input('2) Convert Ch9 radiance to brightness temperature... ')
    x = c9b.bright(wavelength=10.79)
    c9b.disp(window=11)

    if method == 1:
        
        #input('3) Read in and display visible (ch1) image... ')
        c1b = Image('msg_c01_z{}.img'.format(hour_str))  # Load visible channel image
        #c1b.disp(5) 
    
        #input('4) Set threshold values for daytime cloud detection...')
        c9cld = 275
        c1cld = 0.5*55*(1+np.cos((2*np.pi*(hour-12))/24))
        print(c1cld)
        c9_2cld = 293
        
        # Displaying cloud filter parameters on scatter graph:
        c1vec = c1b.data.flatten()
        c9vec = c9b.data.flatten()
        
        # Remove 0 values:
        c1vec_nonzero = c1vec[np.nonzero(c1vec)]
        c9vec_nonzero = c9vec[np.nonzero(c9vec)]
        
        plt.figure(15)
        plt.clf()
        plt.title('C9 v C1  Scatter plot')
        plt.xlabel('C1 Radiance')
        plt.ylabel('C9 Temperature')
        plt.scatter(c1vec,c9vec,s=1,color='black')
        
        plt.plot([min(c1vec),c1cld], [c9cld,c9cld],color='red')
        plt.plot([c1cld,c1cld],[c9cld,c9_2cld],color='red')
        plt.text(0,290,'Cloud Free',color='red')
        plt.plot([c1cld,max(c1vec)], [c9_2cld,c9_2cld],color='red')
        plt.text(280,320,'Cloud Free',color='red')
        plt.xlim(min(c1vec_nonzero)-5,max(c1vec)+5)
        plt.ylim(min(c9vec_nonzero)-5,max(c9vec)+5)
        plt.show()
        
        #input('5) Replot images with daytime cloud mask applied... ')
        # True = cloud-free, False = cloudy
        c1mask = c1b.data.__lt__(c1cld)
        c9mask = c9b.data.__gt__(c9cld)
        cloudmask_1 = np.logical_and(c1mask,c9mask)
        c1mask_2 = c1b.data.__gt__(c1cld)
        c9mask_2 = c9b.data.__gt__(c9_2cld)
        cloudmask_2 = np.logical_and(c1mask_2, c9mask_2)
        cloudmask = np.logical_or(cloudmask_1, cloudmask_2)
        c1b.data = np.where(cloudmask,c1b.data,0) # Set cloudy data to Rad=0
        c1b.title = 'Cloud mask applied to Vis image'
        #c1b.disp(window=7)
        c9b.data = np.where(cloudmask,c9b.data,200) # Set cloudy data to T=200K
        c9b.title = 'Cloud mask applied to Ch9 image'
        c9b.disp(window=12)
        
        image.data = np.where(cloudmask,image.data,0) # Set cloudy data to Rad=0
        image.title = 'Cloud mask applied to input image'
        #image.disp(window=8)
    
        return(image)

    if method == 0:
    
        input('3) Read in and display IR (ch4) image... ')
        c4b = Image('msg_c04_z{}.img'.format(hour_str))# Load visible channel image
        c4b.disp(window=5)            # display image in a different window
    
        input('4) Select threshold values for nighttime cloud detection... ')
        c9cld = temp_threshold # temperature threshold, below which cloud classification
        c4cld = rad_threshold # radiance threshold, below which cloud classification
        c9cld_base = 0 # trivial lower threshold
        c4cld_base = 0 # trivial lower threshold
        left_x, right_x = plt.xlim()
        bottom_y, top_y = plt.ylim()
        
        #plt.plot([left_x,right_x], [c9cld,c9cld],color='red')
        #plt.plot([c4cld,c4cld],[bottom_y,top_y],color='red')
        #plt.text(0,290,'Cloud Free',color='red')
        #plt.text(0.8,290,'Cloud Free',color='red')
        #plt.text(0.8,150,'Cloud Free',color='red')
        #plt.show()
        
        # input('4.5) Create scatter plot of Ch9 v Ch4... ')
        # plt.figure(6)
        # plt.clf()
        # plt.title('C9 v C4  Scatter plot')
        # plt.xlabel('C4 Radiance')
        # plt.ylabel('C9 Temperature')
        # plt.scatter(c4vec,c9vec,s=1,color='black')
        # #plt.scatter(c1vec[idx],c9vec[idx],s=1)
        # plt.show()
        
        input('Replot images with nighttime cloud mask applied... ')
        # True = cloud-free, False = cloudy
        c4mask = c4b.data.__gt__(c4cld)
        c9mask = c9b.data.__gt__(c9cld)
        cloudmask_1 = np.logical_and(c4mask,c9mask)
        #c4mask_2 = c4b.data.__gt__(c4cld)
        #c9mask_2 = c9b.data.__gt__(c9cld_base)
        #cloudmask_2 = np.logical_and(c4mask_2, c9mask_2)
        #cloudmask = np.logical_or(cloudmask_1, cloudmask_2)
        cloudmask = cloudmask_1
        c4b.data = np.where(cloudmask,c4b.data,0) # Set cloudy data to Rad=0
        c4b.title = 'Cloud mask applied to Ch4 image'
        c4b.disp(window=7)
        c9b.data = np.where(cloudmask,c9b.data,200) # Set cloudy data to T=200K
        c9b.title = 'Cloud mask applied to Ch9 image'
        c9b.disp(window=6)
        
        image.data = np.where(cloudmask,image.data,0) # Set cloudy data to Rad=0
        image.title = 'Cloud mask applied to input image'
        image.disp(window=8)
        
        return(image)

      
    
# ------------------------------------------------------------------------------
class Geo:
  """ Geometric calibration data and methods

  DATA 
    cal    boo : True = Geoetric calibration set
    alpha  flt : y/elevation scale factor
    beta   flt : x/azimuth   scale factor
    x0     flt : x-coordinate of sub-satellite point
    y0     flt : y-coordinate of sub-satellite point
    geofil str : Name of file containing GeoCal data (eg 'geo.txt')

 METHODS
    __init__ : Initialise new Geo object   
      menu   : Screen menu for GeoCal part of practical
      save   : Write GeoCal data to file     
      input  : Read user-input of GeoCal parameters 
      angles : Print Elevation,Azimuth angles for givem Lat,Lon
      locang : Convert ele,azi angles to lat,lon,zen angles
      locate : Convert ix,iy coords to lat,lon,zen angles
      satang : Convert lat,lon angles to ele,azi,zen angles
      coords : Convert lat,lon angles to ix,iy coords

  USAGE
    Called once at the start to initialise a Geo object

  HISTORY
    v16Oct20 : AD Original version
  """
 
  def __init__(self,geofil):
    """ Initialise new Geo object

    PARAMETERS
      geofil str : name of file containing geo.cal data, eg 'geo.txt' 

    DESCRIPTION
      If the Geo Cal data file exists, the data are read on initialisation
      and geo.cal set True
    """

    # Local constants
    self.DIST = 42260.0      # Radial dist [km] of sat. from centre of earth
    self.REARTH  = 6371.0    # Earth radius [km]

    try:                     # if file already exists ...
      f = open(geofil,"r")
      rec = f.readline()  
      rec = f.readline()
      flds = rec.split()
      self.y0    = float(flds[0])   # y-coordinate of sub-satellite point
      self.x0    = float(flds[1])   # x-coordinate of sub-satellite point
      self.alpha = float(flds[2])   # y/elevation scale factor
      self.beta  = float(flds[3])   # x/azimuth   scale factor
      f.close()
      self.cal = True               # Flag for GeoCal data set
      print(" *** GeoCal data loaded from file: " + geofil)
    except:                         # file doesn't exist or can't be read
      print(" *** GeoCal data file not found/read: " + geofil)

  def locang(self,ele,azi):
    """ Convert ele,azi angles to lat,lon,zen angles

    PARAMETERS
      ele flt : Elevation angle [deg]
      azi flt : Azimuth angle [deg]

    RETURNS
      lat flt : Latitude [deg N]
      lon flt : Longitude [deg E]
      zen flt : Zenith angle [deg]

    DESCRIPTION
      The inverse of SATANG.
      Uses spherical coordinate geometry to find the point of intersection of a 
      ray leaving the satellite at particular ele,azi angle with the earth 
      surface
      If no intersection, returns (np.nan,np.nan,np.nan) instead.
    """
    rele     = math.radians(ele)
    sinele   = math.sin(rele)
    cosele   = math.cos(rele)
    razi     = math.radians(azi)
    sinazi   = math.sin(razi)
    cosazi   = math.cos(razi)
    # Distance of plane of intersection from centre of earth
    h = self.DIST * sinele 
    if abs(h) > self.REARTH: return (np.nan,np.nan,np.nan)  # no sfc intersect.
    r1 = math.sqrt( self.REARTH**2 - h**2 ) # Radius of circle of intersection
    d1 = self.DIST * cosele
    if abs ( d1 * sinazi ) > r1: return (np.nan,np.nan,np.nan) # No intersection
    # Distance of line of sight
    x = d1 * cosazi - math.sqrt( r1**2 - d1**2 * sinazi**2 )
    # Distance from pixel to point of intersection of earth's vertical axis with
    # plane of intersection
    d2 = self.DIST / cosele
    y = x**2 + d2**2 - 2 * x * d2 * cosazi
    if y < 0.0: y = 0.0
    y = math.sqrt(y)
    h1 = self.DIST * math.tan(rele)
    if abs(h1) > 1.0e-10:     
      gamma = math.acos( ( self.REARTH**2 + h1**2 - y**2 ) / 
                         ( 2.0 * self.REARTH * h1 ) ) 
    else:
      gamma = math.pi / 2.0 - h1 / ( 2.0 * self.REARTH )
    rlat = math.pi / 2.0 - gamma
    gamma1 = math.asin ( sinazi * x / y )
    rlon = math.atan ( math.sin(gamma1) / ( math.cos(gamma1) * cosele ) )
    rzen = math.acos ( cosazi * cosele ) + \
           math.acos ( math.cos(rlat) * math.cos(rlon) )
    lat = math.degrees(rlat)
    lon = math.degrees(rlon)
    zen = math.degrees(rzen)
    return ( lat, lon, zen )

  def locate(self,ix,iy):
    """ Convert ix,iy coords to lat,lon,zen angles

    PARAMETERS
      ix int : Current pixel x-coordinate
      iy int : Current pixel y-coordinate 

    RETURNS
      lat flt : Latitude [deg N]
      lon flt : Longitude [deg E]
      zen flt : Zenith angle [deg]

    DESCRIPTION 
      Uses GeoCal parameters to convert x,y to azi,ele angles 
      then uses locang to convert azi,ele to lat,lon,zen
      If GeoCal has not been set, returns (np.nan,np.nan,np.nan).
      May also return np.nan from locang if x,y beyond edge of disk
    """

    if self.cal:
      ele = ( iy - self.y0 ) / self.alpha
      azi = ( ix - self.x0 ) / self.beta
      return self.locang(ele,azi)
    else:
      return ( np.nan, np.nan, np.nan )

  def satang(self,lat,lon):
    """ Convert lat,lon angles to ele,azi,zen angles

    PARAMETERS
      lat flt : Latitude [deg N]
      lon flt : Longitude [deg E]

    RETURNS
      ele flt : Elevation angle [deg]
      azi flt : Azimuth angle [deg]
      zen flt : Zenith angle [deg]

    DESCRIPTION
      The inverse of locang
      Alpha is elevation and beta angle of rotation about inclined axis measured
      from the central vertical. Assumes spherical earth.
      Inputs/outputs in degrees, although internally converted to radians
    """

    # Convert lat,lon from degrees to radians
    rlat  = math.radians(lat)
    rlon  = math.radians(lon)
    # Height [km] of pixel above horizontal
    h2    = self.REARTH * math.sin(rlat)     
    # Distance [km] from earth's vertical axis
    r2    = self.REARTH * math.cos(rlat)     
    # Horizontal distance of pixel from satellite
    d3    = math.sqrt ( self.DIST**2 + r2**2 - 
                        2 * self.DIST * r2 * math.cos(rlon) )
    delta = math.atan ( h2 / d3 ) 
    gamma = math.asin ( r2 * math.sin(rlon) / d3 )
    rele  = math.atan ( math.tan(delta) / math.cos(gamma) )
    razi  = math.asin ( math.cos(delta) * math.sin(gamma) )
    rzen  = math.acos ( math.cos(razi)  * math.cos(rele) ) + \
            math.acos ( math.cos(rlat)  * math.cos(rlon) )         
    ele = math.degrees(rele)
    azi = math.degrees(razi)
    zen = math.degrees(rzen)
    return ( ele, azi, zen )
    
  def coords(self,lat,lon):
    """ Convert lat,lon angles to x,y coords
    (Intended to be the inverse of the 'locate' function
#### Has been added as an additional function ####

    PARAMETERS
      lat flt : Latitude [deg N]
      lon flt : Longitude [deg E]

    RETURNS
      ix int : Pixel x-coordinate
      iy int : Pixel y-coordinate
      

    DESCRIPTION 
      Uses satang to convert lat, long to azi,ele angles then uses
      GeoCal parameters to convert azi,ele to x,y, rounding them to the nearest integer
    """

    (ele,azi,zen) = self.satang(lat,lon)
    ix = round((self.beta)*azi + self.x0)
    iy = round((self.alpha)*ele + self.y0)

    return(ix,iy)   
 
#-------------------------------------------------------------------------------

# Start of main program

# Define location of initial box?
xmin = 462
xmax = 502
ymin = 420
ymax = 468

print('This program uses a cloud filter to remove clouded areas from the imagery for both day and night. It then uses the noon radiance of the visible channel and the diurnally averaged radiance of the 11um channel to to calculate the coefficients a1 and a9.')
print('The program must be run in the same folder as all of the images from channels 1 (visible); 4 and 9 (IR cloud).')
print('Press <ENTER> to continue through each step\n')

# The below code until [end test] is to test the cloud filter

# turn on interactive mode for plots
plt.ion()  

#input('Read in and display visual image at 1200Z... ')  
#c1b = Image('msg_c01_z12.img')  # Load visible channel image
#c1b.disp(window=1)                      # display image

#input('Apply cloud filter to the image and open it in a new window')
#c1b_cloud_free = c1b.cloud(1,12,280,100)
#c1b_cloud_free.disp(window=2)


#input('Continue if the test was a success')


#input('Load matching Ch9 infrared window image...')
#c9b = Image('msg_c09_z12.img')  # Load Ch9 image
#c9b.disp(window=1)   # display in a different window
#input('Test - remove cloud from this image')
#c9_cloud_free = c9b.cloud(1,12,280,100)
#c9_cloud_free.disp(window=2)


# [end test]


# Load in the image, apply the cloud filter to the initial 1200Z image
input('Load initial channel 1 image and apply daytime cloud filter... ')
c1b = Image('msg_c01_z12.img')  # Load visible channel image
c1b = c1b.cloud(1,12,280,100)

input('Select a region of the visible image to investigate... ')
box={ 'xmin':xmin, 'xmax':xmax, 'ymin':ymin, 'ymax':ymax, 'color':'black' }    #chosen box with limited range of lat/long, near centre longitudinally so at local midday
c1b.disp(window=1,box=box)
#c1b = c1b.clip(box)

input('Create a new image from the selection... ')
c1c = c1b.clip(box,title='Selected part of Vis image over ocean')
c1c.disp()

input('Calculate an average of the radiance in the visible channel image... ')
c1radvec = c1c.data.flatten() # turn it into a single vector of radiance values?

# remove zeros (the cloud pixels)
c1radvec = c1radvec[c1radvec!=0]

R1_ocean = (np.sum(c1radvec))/len(c1radvec)
print('R1 over the ocean is ' + str(R1_ocean) + ' W m^2 sr um.')


input('Initiate the computation of the 24h diurnal average over ocean...')

# Create blank array to store the radiance values for ch9:
c9r9 = np.zeros(24)

# Create loop that will load each image (discard the 0000Z image, consider 24 images total, 1 per hour from 0100Z to 2400Z
for i in range(1,25):
    # Load in the image
    input('Load in  and display the channel 9 image for ' +str(i)+ 'Z')
    if i < 10:
        c9_rad_img = Image('msg_c09_z0' + str(i) +'.img')  # Load the image for the ith hour
    else:
        c9_rad_img = Image('msg_c09_z' + str(i) +'.img')
    
    c9_rad_img.disp() # Comment out for smooth operation
    
    # Call cloud filter to remove cloud from image
    if 8 <= i <= 15:
        c9_rad_img = c9_rad_img.cloud(1,i)
    else:
        c9_rad_img = c9_rad_img.cloud(0,i,285,0.3)
    
    # Select same box area of image
    input('Select the same box area of the image') # Comment out for smooth operation
    c9_rad_img.disp(window=1,box=box) # Comment out for smooth operation
    
    input('Display the box selection for the channel 9 image for ' +str(i)+ 'Z') # Comment out for smooth operation
    c9_rad_box = c9_rad_img.clip(box,title='Selected part of channel 9 image at ' +str(i)+ 'Z')
    c9_rad_box.disp() # Comment out for smooth operation
    
    input('Calculate the average radiance of this box') # Comment out for smooth operation
    # Calculate the average radiance of this box
    c9radvec = c9_rad_box.data.flatten() # turn it into a single vector of radiance values?
    # remove zeros (the cloud pixels) # ASSUMES THE CLOUD FILTER SETS ALL CLOUD TO ZERO RADIANCE
    c9radvec = c9radvec[c9radvec!=0]
    c9r9[i-1] = (np.sum(c9radvec))/len(c9radvec)
    print('At ' +str(i)+ 'Z, R9 over the ocean is ' + str(c9r9[i-1]) + ' W m^2 sr um.')

print(c9r9)  

x = np.linspace(1, 24, num=24)
plt.plot(x,c9r9)

# Below is not actually true: need to plot values and fit to sine curve, then average that over 24h (as doing above)
R9_ocean = (np.sum(c9r9))/len(c9r9)
print('R9 diurnally averaged over the ocean is ' + str(R9_ocean) + ' W m^2 sr um.')


# Box will placed at 2 different locations of ~the same latitude, one on land and one on ocean
# Can do this by choosing open ocean at local noon, then a location on land approximately 1 hour of longitude west, using the 1h earlier (1100Z) photo to ensure local noon is also at that time

# Points to note: restricting latitude to within +- a certain amount in the box - assume that radiance varies approximately with cos(latitude), so area restricted to reduce error?
# (If using a box centred on the equator and extending above and below, then will give an overall underestimate? of radiance due to the cos factor)
# ((Maybe is this covered in some calibration? Possibly not))



# Load Geographical Calibration data
geo = Geo('geo.txt')

#10. Print Lat,Lon of lower left and upper right of box # This will need to move up to the top as it will require the calibration 
(latmin,lonmin,zenmin) = geo.locate(box['xmin'],box['ymin'])
print('Ocean box bottom left corner is at (' +str(latmin)+',' +str(lonmin)+ ')')
(latmax,lonmax,zenmax) = geo.locate(box['xmax'],box['ymax'])
#print(latmax,lonmax)

# If needed, can use the below to calculate the approximate centrepoint of the selection box - may be useful in making moving it 1h west
xcentre = round((box['xmax'] - box['xmin'])/2) + box['xmin']
ycentre = round((box['ymax'] - box['ymin'])/2) + box['ymin']
(latcen,loncen,zencen) = geo.locate(xcentre, ycentre)
#print(latcen,loncen)

# Moving the box 1hr west (to use image 1hr earlier)
(latmin2,lonmin2) = (latmin, lonmin + 15)
(latmax2,lonmax2) = (latmax, lonmax + 15)

# Now need to turn these lat long coordinates back into x,y: (using our own added function)
(xmin2,ymin2) = geo.coords(latmin2,lonmin2)
(xmax2,ymax2) = geo.coords(latmax2,lonmax2)

#print(xmin2,ymin2)

# Define new box using these, and the image from 1h earlier:
input('Read in new image to be used (1hr earlier)')
c1_11 = Image('msg_c01_z11.img')  # Load visible channel image
c1_11.disp(window=1)                      # display image

# Apply cloud filter (sets radiance values at cloud locations to 0)
c1_11 = c1_ll.cloud(1,11,280,100)

# Test box selection 1 hr west
input('Display 1hr west box')
box2={ 'xmin':xmin2, 'xmax':xmax2, 'ymin':ymin2, 'ymax':ymax2, 'color':'black' }
c1_11.disp(window=1,box=box2)

# Create new image
input('Open new box (over land) in a new image')
c1_11selection = c1_11.clip(box2,title='Selected part of Vis image over land')
c1_11selection.disp()

# Print lower left corner of box for verification
(latmin2,lonmin2,zenmin) = geo.locate(box2['xmin'],box2['ymin'])
print('Land box bottom left corner is at (' +str(latmin2)+',' +str(lonmin2)+ ')')

input('Calculate an average of the radiance in the visible channel image... ')
c1_11radvec = c1_11selection.data.flatten() # turn it into a single vector of radiance values?

# remove zeros (the cloud pixels)
c1_11radvec = c1_11radvec[c1_11radvec!=0]

R1_land = (np.sum(c1_11radvec))/len(c1_11radvec)
print('R1 over the land is ' + str(R1_land) + ' W m^2 sr um.')
# Possibly need to multiply this by 1/cos(longitude)?

input('Initiate the computation of the 24h diurnal average over ocean...')

# Create blank array to store the radiance values for ch9:
c9r9_land = np.zeros(24)

# Create loop that will load each image (discard the 0000Z image, consider 24 images total, 1 per hour from 0100Z to 2400Z
for i in range(1,25):
    # Load in the image
    #input('Load in  and display the channel 9 image for ' +str(i)+ 'Z')
    if i < 10:
        c9_land_rad_img = Image('msg_c09_z0' + str(i) +'.img')  # Load the image for the ith hour
    else:
        c9_land_rad_img = Image('msg_c09_z' + str(i) +'.img')
    
    #c9_land_rad_img.disp()
    
    # Conduct cloud filter
    if 7 <= i <= 14:
        c9_land_rad_img = c9_land_rad_img.cloud(1,i,280,100)
    else:
        c9_land_rad_img = c9_land_rad_img.cloud(0,i,285,0.3)

    
    # Select same box area of image
    #input('Select the same box area of the image') # Comment out for smooth operation
    #c9_land_rad_img.disp(window=1,box=box2) # Comment out for smooth operation
    
    #input('Display the box selection for the channel 9 image for ' +str(i)+ 'Z') # Comment out for smooth operation
    c9_land_rad_box = c9_land_rad_img.clip(box2,title='Selected part of channel 9 image at ' +str(i)+ 'Z')
    #c9_rad_box.disp() # Comment out for smooth operation
    
    # Calculate the average radiance of this box
    #input('Calculate the average radiance of this box') # Comment out for smooth operation
    c9_landradvec = c9_land_rad_box.data.flatten() # turn it into a single vector of radiance values?
    
    # remove zeros (the cloud pixels) # ASSUMES THE CLOUD FILTER SETS ALL CLOUD TO ZERO RADIANCE
    c9_landradvec = c9_landradvec[c9_landradvec!=0]
    c9r9_land[i-1] = (np.sum(c9_landradvec))/len(c9_landradvec)
    print('At ' +str(i)+ 'Z, R9 over the land is ' + str(c9r9_land[i-1]) + ' W m^2 sr um.')
    

print(c9r9_land)
# Below is not actually true: need to plot values and fit to sine curve, then average that over 24h
R9_land = (np.sum(c9r9_land))/len(c9r9_land)
print('R9 diurnally averaged over the land is ' + str(R9_land) + ' W m^2 sr um.')
# Possibly need to multiply this by 1/cos(longitude?), as this is per unit pixel area? Not sure


# Calculate the values of a1 and a9:
input('Calculate a1 and a9:')

a9 = (R1_ocean - R1_land)/(R9_land*R1_ocean - R9_ocean*R1_land)

a1 = (1/R1_ocean)*(1-a9*R9_ocean)

print('The value of a1 is ' + str(a1))
print('The value of a9 is ' + str(a9))


input('Press <ENTER> to quit program ...')

