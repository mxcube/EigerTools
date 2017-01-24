"""
   File:  EigerDataSet.py

   Description:  Implements access to an Eiger dataset

   Author:  Proxima2

"""
import math
import re
import h5py
import glob
import math

import os
import sys
import time
import numpy as np

# path to albula python modules
sys.path.insert(0,"/usr/local/xtal/albula/dectris/albula/3.2/python")
try:
    import dectris.albula
    albula_imported = True
except ImportError:
    print ("Cannot find albula library. Some features disabled")
    albula_imported = False

class EigerDataSet(object):

    re_datafile = re.compile("(?P<rootname>.*)_data_\d+.h5")
    re_masterfile = re.compile("(?P<rootname>.*)_master.h5")

    #
    #  An 9M Eiger image is composed of 18 blocks (2modules/each) in a 3x6 array
    #   separated by vertical and horizontal stripes
    #
    #  One image is 3110x3269 pixels 
    #

    # for 16M, it's 32 blocks (2modules/each) in a 4x8 array
    # image size is 4150x4371 pixels
    mod_n_rows = 6
    mod_n_cols = 3

    m_w = 1030 # module width
    m_h = 514 # module height
    h_h = 37 # horizontal strip height
    v_w = 10 # vertical strip width

    h_o = m_w + v_w # horizontal offset between blocks
    v_o = m_h + h_h # vertical offset between blocks

    def __init__(self, filename): 

        self.filename = filename

        d = self.re_datafile.match(filename)
        m = self.re_masterfile.match(filename)

        if d is not None:
            self.rootname = d.group('rootname') 
            self.master_filename = self._master_filename()
            self.master_f = h5py.File(self.master_filename)
        elif m is not None:
            self.rootname = m.group('rootname') 
            self.master_filename = filename
            self.master_f = h5py.File(filename)
        else:
            self.master_f = None
            raise IOError("Filename is not a standard dectris name. Cannot find master file")

        self._init_data()
        self._init_detector()

    def get_master_filename(self):
        return self.master_filename

    def get_rootname(self,basename=True):
        if basename:
            return os.path.basename(self.rootname)
        else:
            return self.rootname

    def beam_center(self):
        if self.master_f is None:  return None

        center_x = self.master_f['/entry/instrument/detector/beam_center_x'].value
        center_y = self.master_f['/entry/instrument/detector/beam_center_y'].value
        return [center_x, center_y]

    def get_number_images(self):
        nb_triggers = self.get_number_triggers()
        nb_images = self.get_images_per_trigger()
        return nb_triggers * nb_images  
        
    def get_images_per_trigger(self):
        if self.master_f is None:  return None

        return self.master_f['entry/instrument/detector/detectorSpecific/nimages'].value

    def get_number_triggers(self):
        if self.master_f is None:  return None

        return self.master_f['entry/instrument/detector/detectorSpecific/ntrigger'].value

    def get_wavelength(self):
        if self.master_f is None:  return None

        return self.master_f['/entry/instrument/beam/incident_wavelength'].value

    def get_detector_distance(self):
        if self.master_f is None:  return None

        return self.master_f['/entry/instrument/detector/detector_distance'].value

    def get_detector_size(self):
        if self.master_f is None:  return None

        size_x = self.master_f['/entry/instrument/detector/detectorSpecific/x_pixels_in_detector'].value
        size_y = self.master_f['/entry/instrument/detector/detectorSpecific/y_pixels_in_detector'].value
        return size_x, size_y

    def get_pixel_size(self):
        if self.master_f is None:  return None

        px_size_x = self.master_f['/entry/instrument/detector/x_pixel_size'].value
        px_size_y = self.master_f['/entry/instrument/detector/y_pixel_size'].value
        return px_size_x, px_size_y

    def get_image_data(self, imgno, apply_mask=True):
        img = self._get_image(imgno)
        if img is not None:
            data =  img.data()
            mask = self.get_image_mask(img)
            if apply_mask:
                self._apply_mask(data,mask)
            return data
        else:
            return None

    def get_image_mask(self, img):
        if not albula_imported: 
            raise NotImplementedError("albula module missing. save thumbnail feature not available")
            return

        # get first image on first file 
        if not self._mask_read: 
            if self._image_info:
                print("Reading image mask")
                self._mask = img.optionalData().pixel_mask().data()
                self._mask_read = True

        return self._mask

    def save_thumbnail(self, binfactor, start_image=0, nb_images=5, output_file=None, inverted=True, rings=None):

        data = self._merge_data(start_image, nb_images)
        mods = self._split_modules(data)

        reduced_modules = []
        for mod in mods:
            reduced_module = self._module_bin(mod, binfactor)
            reduced_modules.append( reduced_module )

        h_h_binned = int(self.h_h / binfactor) # binning of horizontal stripe
        v_w_binned = int(self.v_w / binfactor) # binning of vertical stripe

        reduced_image = self._rebuild_image(reduced_modules, (self.mod_n_cols,self.mod_n_rows), h_h_binned, v_w_binned)
    
        ring_info=[]
        center = [int(cen/binfactor) for cen in self.beam_center()]

        #for distance in (0.25, 0.50, 0.75, 1.00, 1.25):
        for radius in rings:
            resolution = self.resolution_at_radius(radius)
            if resolution is not None:
                factor = [radius, radius]
                label = "%3.2f A" % resolution
                ring_info.append([factor,center,label])
            else:
                print "Cannot determine resolution for ring (%s)" % radius 
     
        self._do_save_thumbnail(reduced_image, output_file, rings=ring_info) 
        print("Thumbnail saved to %s" % os.path.abspath(output_file))

    def resolution_at_radius(self, radius_px):
        """
        description:
    
        args:
            filename:  a dectris series datafile. 
                       filename has to conform to the format <prefix>_data_<setno>.h5
                       and a master file has to exist at the same level as filename with
                       name:  <prefix>_master.h5
    
            radius_px: percentage from center at which to calculate resolution

        returns: [resolution, offsets]
    
            resolution: 
            offsets: [offset_x, offset_y] or None
                  distance (in pixels) from center at which the beam center is, these values
                  are used to estimate the resolution
                  if offsets is None, image center and beam center coincides
     
        """
        if self.master_f is None:  return None

        size_x, size_y = self.get_detector_size() 
        detdist = self.get_detector_distance()
        wavelength = self.get_wavelength()
        px_size_x, px_size_y = self.get_pixel_size()
    
        x_size = size_x * px_size_x
        y_size = size_y * px_size_y

        radius_m = x_size/2.0 * radius_px
        tth = math.atan(radius_m / detdist)
        res = 0.5 * wavelength / math.sin(tth/2.0)

        return res
    
    def _init_detector(self):
        # get detector module, block and detector size, 
	
        #get block size
        m_w, m_h = self.master_f['/entry/instrument/detector/detectorSpecific/detectorModule_000/data_size'].value # module width
        print "modules size is ", m_w, m_h
	self.m_w = m_w
        self.m_h = m_h * 2 # each block is composed of two modules

        self.h_h = 37 # horizontal strip height
        self.v_w = 10 # vertical strip width
        self.h_o = self.m_w + self.v_w # horizontal offset between blocks
        self.v_o = self.m_h + self.h_h # vertical offset between blocks


        #get detector size (needed, esp. with different ROI)
        det_x, det_y = self.get_detector_size()
        self.mod_n_cols = (det_x + self.v_w)/self.h_o
        self.mod_n_rows = (det_y + self.h_h)/self.v_o


    def _init_data(self):
        data_filenames = glob.glob("%s_data_*.h5" % self.rootname)
        data_filenames.sort()
        

        self._mask = np.zeros(0)
        self._mask_read = False
        
        if not albula_imported: 
            return

        series = dectris.albula.DImageSeries() 

        self._image_info = []

        nb_images = 0
        for datafile in data_filenames: 
            series.open(datafile)
            size = series.size()
            self._image_info.append( [datafile, nb_images, size] )
            nb_images += size

        self._total_images = nb_images
        self._check_info()
  
    def _check_info(self):
        self._nb_dataset_images = self.get_number_images()
        if (self._nb_dataset_images != self._total_images): 
            print("Incomplete eiger dataset")
            return False
        else:
            return True
        
    def _master_filename(self):
        _master = "%s_master.h5" % self.rootname 
        return _master

    def _merge_data(self, start_image, nb_images):
        data_merged = None

        for imgno in range(start_image, start_image+nb_images):
            print("Getting image no: %s" % imgno)
            data = self.get_image_data(imgno, apply_mask=True) 
            if data is None:
                print("no data found for image no: %s" % imgno)
                continue

            if data_merged is None:
                data_merged = self._filter_data_low(data)
            else:
                data_merged += self._filter_data_low(data)

        return data_merged

    def _apply_mask(self, data, imgno):
        mask = self.get_image_mask(imgno)
        data[mask.nonzero()] = pow(2,8)-1

    def _filter_data_low(self, data, minval=5):
        data[data<minval] = 0
        return data

    def _filter_data_high(self, data, maxval):
        data[data>maxval] = maxval 
        return data
    
    def _filter_data(self, data, minval, maxval):
        data[data>maxval] = maxval
        data[data<minval] = minval
        return data

    def _split_modules(self, img):
        """
        Separate one eiger image in its composing modules.
        Modules are returned as a list with first module being
        the one at the top left of the image, ordered from
        left to right, then top to bottom
        """
    
        frame_coords = []
    
        for i in range(self.mod_n_rows):
            yoff = self.v_o * i
            for j in range(self.mod_n_cols):
                xoff = self.h_o * j 
    
                xend = xoff+self.m_w; yend=yoff+self.m_h
                frame_coords.append( (xoff, xend, yoff, yend) )
    
        mods = []
        for fr in frame_coords:
            xb, xe, yb, ye = fr
            mod0 = img[yb:ye,xb:xe]
            mods.append(mod0)
    
        return mods
    
    def _module_bin(self, mod, binning, scaling=True):
        """
        calculate
        """
        height, width = mod.shape   
        
        # If binning can be applied in both directions, go fast
        if width % binning == 0 and height % binning == 0:
            e = mod.reshape( height/binning, binning, width/binning, binning)
            return e.sum(axis=3).sum(axis=1)
    
        # If not... 
    
        # First sum columns
        if width % binning == 0:
            # standard binning can be applied
            e = mod.reshape(height, width/binning, binning)
            xbin = e.sum(axis=2)
        else:
            # standard binning can not be applied. sum as much as possible
            # with standard number of rows, then with one extra row for the rest
            # apply or not scaling correction
            add_more = width % binning
            add_bin = width/binning - add_more
    
            vecs = []
            colno = 0
            for i in range(add_bin):  
                vec = mod[:,colno:colno+binning].sum(axis=1)
                colno += binning
                vecs.append(vec)
    
            for i in range(add_more):  
                vec = mod[:,colno:colno+binning+1].sum(axis=1)
                if scaling:
                    vec = np.rint(vec * binning / (binning+1))
                colno += binning+1
                vecs.append(vec)
            xbin = np.column_stack(vecs)
            
        # Then sum rows
        height, width = xbin.shape
    
        if height % binning == 0:
            # standard binning can be applied
            e = mod.reshape(height/binning, binning, width)
            ybin = e.sum(axis=1)
        else:
            # standard binning can not be applied. sum as much as possible
            # with standard number of rows, then with one extra row for the rest
            # apply or not scaling correction
            add_more = height % binning
            add_bin = height/binning - add_more
    
            vecs = []
            rowno = 0
            for i in range(add_bin):  
                vec = xbin[rowno:rowno+binning].sum(axis=0)
                rowno += binning
                vecs.append(vec)
    
            for i in range(add_more):  
                vec = xbin[rowno:rowno+binning+1].sum(axis=0)
                if scaling:
                    vec = np.rint(vec * binning / (binning+1))
                rowno += binning+1
                vecs.append(vec)
            ybin = np.row_stack(vecs)
            
        return ybin
    
    def _rebuild_image(self, mods, geometry, stripe_height, stripe_width, stripe_value=0):
        """
        Rebuilds a data image from a series of composing modules, placing 
        stripes between them. 
    
           mods :  list of modules of same shape
           geometry:  list of   [ mods-per-row , mods-per-column ]
               mods-per-row x mods-per-column must be equal to len(mods)
           stripe_height: height of horizontal stripes
           stripe_width: width of vertical stripes
           stripe_value (optional):  set stripes data value (0 default)
        """
        mod_height, mod_width = mods[0].shape
    
        nb_mods_row, nb_mods_column = geometry
    
        # vertical segment with the height of one module 
        vertical_segment = np.array([stripe_value,] * mod_height * stripe_width)
        vertical_segment = vertical_segment.reshape(mod_height, stripe_width)
     
        # horizontal segments will span over a full row
        hor_width = nb_mods_row * mod_width + stripe_width * (nb_mods_row - 1)
        horizontal_segment = np.array([stripe_value,] * hor_width * stripe_height)
        horizontal_segment = horizontal_segment.reshape( stripe_height, hor_width)
    
        total_height = mod_height * nb_mods_column + stripe_height * (nb_mods_column-1)
        total_width = hor_width
    
        modno = 0
        one_column_mods = []
    
        for rowno in range(nb_mods_column):  
            one_row_mods = []
            for colno in range(nb_mods_row):  
                mod = mods[modno]; modno+= 1
                one_row_mods.append(mod)
                if colno != (nb_mods_row-1):
                    one_row_mods.append(vertical_segment)
    
            one_row = np.column_stack( one_row_mods )
            one_column_mods.append(one_row)
            if rowno != (nb_mods_column-1):
                one_column_mods.append(horizontal_segment)
    
        img = np.row_stack(one_column_mods)
        return img
           
    def _do_save_thumbnail(self, data, outfile=None, inverted=True, rings=None):
        if not albula_imported: 
            raise NotImplementedError("albula module missing. save thumbnail feature not available")
            return

        from PIL import Image
        from PIL.ImageOps import invert
    
        size = data.shape
        pimg = Image.fromarray(data.astype(np.uint8))

        if inverted:
           pimg = invert(pimg)  

        if rings is not None:
            for ring in rings:
                radius, offset, label = ring
                self._draw_ring(pimg, radius, offset, label=label)
    
        pimg.thumbnail(size, Image.ANTIALIAS)
        pimg.save(outfile, "JPEG")

    def _get_image(self, imgno):
        if self._image_info:
             series = dectris.albula.DImageSeries()
             for info in self._image_info:
                 dfile, first_img, size = info
                 if imgno >= first_img and imgno < first_img+size:
                     series.open(dfile)
                     no_in_file = imgno + 1
                     print "   / reading imgno %s from %s (size: %s)" % (no_in_file, dfile, size)
                     return series[no_in_file] 
             else:
                 print("Cannot find image in dataset")
        
        return None

    def _draw_ring(self, image, radius, offset=None, color="#ff9999", label=None):
        """
        Draw a ring in an image object
    
        Params:
            image:  image object
            radius: tuple (as percentage of image size)
            offset: coordinates of image center (as percentage of image size) default = (0.5, 0.5)
        """
        from PIL import Image
        from PIL import ImageDraw
    
        width, height = image.size
    
        rad_x = width * radius[0] / 2.0
        rad_y = height * radius[1] / 2.0
    
        if offset is None:
            offset = [width*0.5,height*0.5]
    
        off_x = offset[0]
        off_y = offset[1]
    
        beg_x = int(off_x - rad_x )
        end_x = int(off_x + rad_x )
        beg_y = int(off_y - rad_y )
        end_y = int(off_y + rad_y )
    
        cbox = (beg_x, beg_y, end_x, end_y)
    
        draw = ImageDraw.ImageDraw(image)
        draw.ellipse(cbox,outline=color)
    
        if label:
            angle = math.radians(45)
            pos_x = off_x + rad_x * math.cos(angle)
            pos_y = off_y - rad_y * math.sin(angle)
            draw.text((pos_x,pos_y), label, fill=color )
    
"""
    This module builds a thumbnail from an hdf5 file containing
    a series of Eiger images

    It does:
      - open hdf5, reads nb_images, sum them up
      - separate every module (without the stripes between modules)
      - reduce the data in every module by applying a binning
      - build a full data image with those reduced modules and set stripes between them
      - generates a jpeg image from the full (reduced) data and saved it
"""

def main():
    import sys

    input_file = sys.argv[1]
    binfactor = int(sys.argv[2])

    rootname, ext = os.path.splitext(input_file)

    output_file = rootname + "_bin%02d.jpg" % binfactor

    default_nb_images = 5

    if len(sys.argv) >= 4:
        nimages = int(sys.argv[3])
    else:
        nimages = default_nb_images

    dataset = EigerDataSet(input_file)

    print "Beam Center (px):", dataset.beam_center()
    print "Number of images:", dataset.get_number_images()
    print "Number of triggers:", dataset.get_number_triggers()
    print "Resolution (radius 0.5):", dataset.resolution_at_radius(0.5)

    rings = [0.25, 0.50, 0.75, 1.00, 1.25]
    dataset.save_thumbnail(binfactor, output_file=output_file, nb_images=nimages, rings=rings) 

if __name__ == '__main__':
    main()
