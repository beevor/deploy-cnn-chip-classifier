# gbdx.Task('deploy-classifier', chips + geoj, model, classes, max_pixel_intensity, min_side_dim, max_side_dim)
import logging
import ast, os, time
import geojson, json
import numpy as np

from keras.models import load_model
from mltools import geojson_tools as gt
from shutil import copyfile
from functools import partial
from multiprocessing import Pool, Process, Queue, cpu_count
from scipy.misc import imresize
from osgeo import gdal
from gbdx_task_interface import GbdxTaskInterface

# log file for debugging
logging.basicConfig(filename='out.log',level=logging.DEBUG)

def check_chip(feature, min_side_dim, max_side_dim):
    '''
    delete a chip if it is too large or small. This must take place outside of the class
        to be performed in parallel.
    '''
    chip_name = str(feature['properties']['feature_id']) + '.tif'

    # Determine side dims
    try:
        chip = gdal.Open(chip_name)
        min_side = min(chip.RasterXSize, chip.RasterYSize)
        max_side = max(chip.RasterXSize, chip.RasterYSize)

    except (AttributeError, RuntimeError):
        logging.debug('Chip not found in directory: ' + chip_name)
        return True

    # Close chip
    chip = None

    # Remove chip if too small or large
    if min_side < min_side_dim or max_side > max_side_dim:
        os.remove(chip_name)


def get_chip(chip_name_num, input_shape, norm_val, max_side_dim, min_side_dim):
    '''
    Get raster array from a chip given the chip name. This is outside the class to allow
        parallel performance

    chip_name_num: [chip_name, order it should appear in list (int)]
    '''
    chip, order = chip_name_num

    # Create normed raster array
    ds = gdal.Open(chip)
    raster_array = []

    for n in xrange(1, input_shape[0] + 1):
        raster_array.append(ds.GetRasterBand(n).ReadAsArray() / norm_val)

    # pad to input shape
    chan, h, w = np.shape(raster_array)
    pad_h, pad_w = max_side_dim - h, max_side_dim - w
    chip_patch = np.pad(raster_array, [(0, 0), (pad_h/2, (pad_h - pad_h/2)),
                        (pad_w/2, (pad_w - pad_w/2))], 'constant',
                        constant_values=0)

    # resize chip if necessary
    if max_side_dim != input_shape[-1]:
        new_chip = []
        for band_ix in xrange(len(chip_patch)):
            new_chip.append(imresize(chip_patch[band_ix],
                            input_shape[-2:]).astype(float))
        chip_patch = np.array(new_chip)

    # Return raster array and order
    return chip_patch, order


class DeployClassifier(GbdxTaskInterface):

    def __init__(self):
        '''
        Instantiate string and data inputs, organize data for training
        '''
        GbdxTaskInterface.__init__(self)
        self.check_chip = check_chip
        self.get_chip = get_chip
        self.start = time.time()

        # Get string inputs
        self.classes = self.get_input_string_port('classes', default=None)
        self.max_pixel_intensity = float(self.get_input_string_port('max_pixel_intensity', default='8'))
        self.min_side_dim = int(self.get_input_string_port('min_side_dim', default='0'))
        self.max_side_dim = ast.literal_eval(self.get_input_string_port('max_side_dim',
                                                                        default='None'))

        # Get input directories
        self.chip_dir = self.get_input_data_port('chips')
        self.model_dir = self.get_input_data_port('model')

        # Format classes: ['class_1', 'class_2']
        if self.classes:
            self.classes = [clss.strip() for clss in self.classes.split(',')]
        else:
            with open(os.path.join(self.model_dir, 'info', 'classes.json')) as f:
                self.classes = ast.literal_eval(json.load(f)['classes'])

        # Get files in input directories
        self.geojson = [f for f in os.listdir(self.chip_dir) if f == 'ref.geojson'][0]
        self.chips = [img for img in os.listdir(self.chip_dir) if img.endswith('.tif')]
        self.model = [mod for mod in os.listdir(self.model_dir) if mod.endswith('.h5')][0]
        self.model = os.path.join(self.model_dir, self.model)

        # Format working directory (chip_dir)
        copyfile(self.model, os.path.join(self.chip_dir, 'model.h5'))
        os.chdir(self.chip_dir)             #!!! Now in chip_dir for remainder of task !!!
        self.model = 'model.h5'

        # Get output directory
        self.outdir = self.get_output_data_port('classified_geojson')
        os.makedirs(self.outdir)
        print 'Took {} seconds to set instance variables'.format(str(time.time() - self.start))


    def filter_chips(self):
        '''
        Remove chips that are too large or too small. Remove entries in ref.geojson that
            are not valid chips.
        This method should be called from the chips directory, which contains the
            reference geojson for each chip.
        '''
        # Open reference geojson
        with open(self.geojson) as f:
            data = geojson.load(f)
            feature_collection = data['features']
        logging.info(str(len(self.chips)) + ' chips in directory before filtering')

        # Remove invalid chips in parallel
        check = partial(self.check_chip, min_side_dim=self.min_side_dim,
                        max_side_dim=self.max_side_dim)
        p = Pool(cpu_count())
        p.map(check, feature_collection)
        p.close()
        p.join()

        # Remove geojson entries with invalid chips
        chip_ids = [f[:-4] for f in os.listdir('.') if f.endswith('.tif')]
        valid_feats = []
        for feat in feature_collection:
            if str(feat['properties']['feature_id']) in chip_ids:
                valid_feats.append(feat)
        logging.info(str(len(chip_ids)) + ' chips remaining after filtering.')
        logging.info(str(len(valid_feats)) + ' chips referenced in filtered geojson.')

        # Return valid geometries
        data['features'] = valid_feats
        with open('target.geojson', 'wb') as f:
            geojson.dump(data,f)

        return 'target.geojson'


    def get_chips_from_features(self, feature_collection, input_shape):
        '''
        Load chips into memory from a list of features in parallel
        Each chip will be padded to the input side dimension
        '''
        chip_names = []

        # Get chip names for each feature
        for ct, feat in enumerate(feature_collection):
            name = str(feat['properties']['feature_id']) + '.tif'
            chip_names.append([name, ct])

        get = partial(self.get_chip, input_shape=input_shape, norm_val=self.max_pixel_intensity,
                      min_side_dim=self.min_side_dim, max_side_dim=self.max_side_dim)
        p = Pool(cpu_count())
        chips = p.map(get, chip_names)
        p.close()
        p.join()

        # Sort arrays to maintain original order of features
        chips.sort(key=lambda tup: tup[1])
        X = [ch[0] for ch in chips]

        return np.array(X)


    def deploy_model(self, model, target_geojson):
        '''
        deploy the model on a feature collection. saves a geojson with
        '''
        yprob, out_file = [], 'classified.geojson'
        input_shape = model.input_shape[-3:]

        # Get features
        with open(target_geojson) as f:
            data = geojson.load(f)
            features = data['features']

        # Classify chips in batches of 1000
        for ix in xrange(0, len(features), 1000):
            this_batch = features[ix: (ix + 1000)]

            get_chip_time = time.time()
            X = self.get_chips_from_features(this_batch, input_shape)
            print 'Took {} seconds to get chips'.format(str(time.time() - get_chip_time))

            deploy_time = time.time()
            yprob += list(model.predict_proba(X))
            print 'Took {} seconds to deploy on chips'.format(str(time.time() - deploy_time))

        # Get predicted classes and certainty
        yhat = [self.classes[np.argmax(i)] for i in yprob]
        ycert = [round(float(np.max(j)), 10) for j in yprob]

        # Update geojson, save as output_name
        data = zip(yhat, ycert)
        property_names = ['CNN_class', 'certainty']
        gt.write_properties_to(data, property_names=property_names,
                               input_file=target_geojson, output_file=out_file)

        return out_file


    def invoke(self):
        '''
        Execute task
        '''
        # Load model
        mod_time = time.time()
        model = load_model(self.model)
        print 'Took {} seconds to load model'.format(str(time.time() - mod_time))

        # Filter chips and geojson
        filter_time = time.time()
        target_geojson = self.filter_chips()
        print 'Took {} seconds to filter chips'.format(str(time.time() - filter_time))

        # Deploy model
        out_file = self.deploy_model(model, target_geojson)
        copyfile(out_file, os.path.join(self.outdir, 'classified.geojson'))


if __name__ == '__main__':
    with DeployClassifier() as task:
        task.invoke()
