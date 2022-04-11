# import os
from gc import collect
# from glob import glob
from struct import unpack
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, use
plt.rcParams.update({'font.size': 14})
use('Agg')

# feel free to add to this
# follow the following...
# how many um are in a m -> 1e6 and so on
_unit_list = {
    'um': 1e6, 'm': 1, 'nm': 1e9,
    'ns': 1e9, 'us': 1e6, 'ps': 1e12
}

# ██████  ███████  █████  ██████
# ██   ██ ██      ██   ██ ██   ██
# ██████  █████   ███████ ██   ██
# ██   ██ ██      ██   ██ ██   ██
# ██   ██ ███████ ██   ██ ██████

# ███████ ██████  ███████
# ██      ██   ██ ██
# ███████ ██   ██ █████
#      ██ ██   ██ ██
# ███████ ██████  ██

class ReadSDF():
    """
    A class for reading in SDF files and plotting the outputs
    A bit longer description.

    Args:
        file (string): SDF file name
        supress (boolean): True silences extra print statements
    """

    def __init__(self, file, supress=True):
        self.fname = file
        self.supress = supress
        self.f = open(self.fname, 'rb')
        self.read_header()
        self.block_locations = {}
        self.get_block_locations()

    def close_file(self):
        # Close the file
        self.f.close()

    def read_header(self):
        """
        Reads the header of SDF file

        Information correspronding to the layout and contents of the sdf file
        is stored and some basic simulation info
        """

        head = self.f.read(106)
        if byte_to_int(head[4:8]) != 16911887:
            raise ValueError(
                'File was written on a big endian machine!', byte_to_int(head[4:8]))

        self.sdf_magic = byte_to_str(head[0:4])
        self.version = byte_to_int(head[8:12])
        self.revision = byte_to_int(head[12:16])
        self.code_name = byte_to_str(head[16:48])
        self.next_block = byte_to_int(head[48:56])
        self.summary_block = byte_to_int(head[56:64])
        self.summary_size = byte_to_int(head[64:68])
        self.nblocks = byte_to_int(head[68:72])
        self.block_head_len = byte_to_int(head[72:76])
        self.sim_time = byte_to_float(head[80:88])
        self.str_len = byte_to_int(head[96:100])
        # print(self.next_block)
        print(f'Sim Time: {self.sim_time}')

    def read_block_header(self, block_name=None):
        """
        Reads the header of a block in the SDF file

        If no block name is given the function will read the block that is next.
        Each the time a header is read the next block location is recorded.

        Args:
            block_name (str): Name of data block to read
        Returns:
            flag (bool): Whether all headers have been added to the block\
                        location dictionary
        """
        # continue from previous location or start at specific point
        if block_name == None:
            position = self.next_block
        else:
            position = self.block_locations[block_name][0]

        self.meta_start = self.next_block + self.block_head_len
        self.f.seek(position, 0)
        head = self.f.read(72 + self.str_len)
        self.block_name = byte_to_str(head[68:68 + self.str_len])
        self.data_loc = byte_to_int(head[8:16])
        self.block_id = byte_to_str(head[16:48])
        self.data_len = byte_to_int(head[48:56])
        self.block_type = byte_to_int(head[56:60])
        self.datatype = byte_to_int(head[60:64])
        self.ndims = byte_to_int(head[64:68])  # 1 if not an array

        self.block_info_len = byte_to_int(
            head[68 + self.str_len:72 + self.str_len])
        # add block position to dictionary
        if self.block_name[0] not in self.block_locations.keys():
            flag = False
            self.block_locations[self.block_name[0]] = [
                self.next_block, self.block_type]
        else:
            flag = True
        # get position of next block
        self.next_block = byte_to_int(head[0:8])
        return flag

    def get_block_locations(self):
        """
        Get location of blocks in SDF file
        If not supressed - file contents is printed

        Args:
            self (object): ReadSDF class object
        """
        #print('Finding what data is stored and where it is...')
        while not self.read_block_header():
            continue
        if not self.supress:
            print("\nAvailable blocks:")
            for v in self.block_locations.keys():
                print(v)
            print('\n')

    def get_file_contents(self):
        """
        Get a list of blocks in SDF file
        Args:
            self (object): ReadSDF class object
        Returns:
            (list): List of block names
        """
        return list(sdf.block_locations.keys())

    def get_variable(self, block_name, **kwargs):
        """
        Calls correct function to get requested variable

        Args:
            block_name (str): Name of data that is required

        Returns:
            various: returned data depends on what is requested

        """
        block_type = self.block_locations[block_name][1]
        if block_type == 1:
            return self.get_plain_mesh(block_name, **kwargs)
        elif block_type == 3:
            return self.get_plain_variable(block_name, **kwargs)

    def get_plain_mesh(self, block_name=None, **kwargs):
        if block_name == None:
            position = self.meta_start
        else:
            position = self.block_locations[block_name][0] + \
                self.block_head_len
            self.read_block_header(block_name)

        self.f.seek(position)
        n = self.ndims
        meta = self.f.read(92 * n + 4)

        mults = byte_to_float(meta[0:8 * n])
        labels = byte_to_str(meta[8 * n:40 * n])
        units = byte_to_str(meta[40 * n:72 * n])
        geo_type = byte_to_int(meta[72 * n:72 * n + 4])
        if geo_type == 0:
            raise Exception('Unknown geometry type!')
        minval = byte_to_float(meta[72 * n + 4:80 * n + 4])
        maxval = byte_to_float(meta[80 * n + 4:88 * n + 4])

        dims = byte_to_int(meta[88 * n + 4:92 * n + 4], split=n)

        data = []
        offset = self.data_loc
        type_size = self.data_len // np.sum(dims)

        for i in range(self.ndims):
            data.append(np.memmap(self.fname, dtype=self._get_data_type(), mode='r',
                                  offset=offset, shape=dims[i], order='F'))
            offset += dims[i] * type_size

        args = {'labels': labels, 'units': units, 'mults': mults}

        return data, args

    def get_plain_variable(self, block_name=None, **kwargs):
        get_grid = False if 'get_grid' not in kwargs.keys(
        ) else kwargs['get_grid']

        if block_name == None:
            position = self.meta_start
        else:
            position = self.block_locations[block_name][0] + \
                self.block_head_len
            self.read_block_header(block_name)

        if sdf.block_type != 3:
            raise ValueError(
                'Expected block_type 3, got block_type', self.block_type)
        self.f.seek(position)
        meta = self.f.read(88)
        mults = byte_to_float(meta[0:8])
        if mults != 1:
            raise Exception(
                'mults != 1, figure out what to do... mults=', mults)

        units = byte_to_str(meta[8:40])
        mesh_id = byte_to_str(meta[40:72])
        npoints = []
        for i in range(self.ndims):
            npoints.append(byte_to_int(meta[72 + i * 4:76 + i * 4]))

        self.stagger = byte_to_int(
            meta[72 + 4 * self.ndims:76 + 4 * self.ndims])
        # if stagger != 0:
        #     raise Exception('stagger != 0, figure out what to do... stagger=', stagger)

        data = np.memmap(self.fname, dtype=self._get_data_type(), mode='r',
                         offset=self.data_loc, shape=np.prod(npoints), order='F')
        data = data.reshape(npoints[::-1])

        data_args = {'labels': block_name, 'units': units, 'mults': mults}

        if get_grid:
            if mesh_id[0] == 'grid':
                #print("Trying to get 'Grid/Grid'")
                grid, grid_args = self.get_plain_mesh(block_name='Grid/Grid')
            else:
                print("mesh_id is", mesh_id[0], "I don't know what to do...")
            return (grid, grid_args), (data, data_args)
        else:
            return (data, data_args)

        # ██████  ██       ██████  ████████
        # ██   ██ ██      ██    ██    ██
        # ██████  ██      ██    ██    ██
        # ██      ██      ██    ██    ██
        # ██      ███████  ██████     ██

    def make_cmap(self, grid, grid_args, data, data_args, config, fnum, fldr):
        config = config.copy()
        grid = grid[:]
        data = data[:]

        for i, u in enumerate(grid_args['units']):
            if u == 'm':
                grid[i] = grid[i] * _unit_conv(config['spatial_scale'])
            else:
                raise ValueError('Unknown spatial unit', grid_args['units'])
        if config['spatial_scale'] == 'um':
            config['spatial_scale'] = '$\mu$m'

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set(
            xlabel=f"{grid_args['labels'][0]} [{config['spatial_scale']}]",
            ylabel=f"{grid_args['labels'][1]} [{config['spatial_scale']}]",
        )
        title = f"{data_args['labels']}\n\
                {self.sim_time * _unit_conv(config['time_unit']):.2f} \
                {config['time_unit']}\n"
        ax.set_title(title, loc='left')
        if data[::-1][::10, ::10].min() > 0:
            im = ax.pcolormesh(
                grid[0][::10], grid[1][::10],
                data[::-1][::10, ::10], shading='auto',
                norm=colors.LogNorm())
        else:
            im = ax.pcolormesh(
                grid[0][::10], grid[1][::10],
                data[::-1][::10, ::10], shading='auto')
        cbar = fig.colorbar(im)
        cbar.ax.set_ylabel(f"${data_args['units'][0]}$", size=20)
        fname = data_args['labels'].split('/')
        fname = '_'.join(fname)
        plt.savefig(f"img/readsdf_out/{fldr}/cmap/{fnum}_{fname}.png")
        plt.close('all')
        collect()

    def make_position_plot(self, grid, grid_args, data, data_args, config, fnum, fldr, axis=0):
        config = config.copy()
        grid = grid[:]
        data = data[:]

        for i, u in enumerate(grid_args['units']):
            if u == 'm':
                grid[i] = grid[i] * _unit_conv(config['spatial_scale'])
            else:
                raise ValueError('Unknown spatial unit', grid_args['units'])
        if config['spatial_scale'] == 'um':
            config['spatial_scale'] = '$\mu$m'

        plot_data = np.mean(data, axis=axis)
        pos = grid[axis]
        pos = np.linspace(pos.min(), pos.max(), len(plot_data))

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set(
            xlabel=f"{grid_args['labels'][0]} [{config['spatial_scale']}]",
            ylabel=f"{_format_string(data_args['labels'])} [{data_args['units'][0]}]",
        )

        title = f"{data_args['labels']}\n\
                {self.sim_time * _unit_conv(config['time_unit']):.2f} \
                {config['time_unit']}\n"
        ax.set_title(title, loc='left')

        ax.plot(pos, plot_data, '-')

        fname = data_args['labels'].split('/')
        fname = '_'.join(fname)
        pos_str = 'x' if axis == 0 else 'y'
        plt.savefig(f"img/readsdf_out/{fldr}/x/{fnum}_{pos_str}_{fname}.png")
        plt.close('all')
        collect()

    # The following functions should not be used
    # These are just used in the prewritten functions above

    def _get_data_type(self):
        """
        Returns the datatype of data in a given block. Not used by end user
        """
        dtype = {0: 'null', 1: 'int32', 2: 'int64', 3: 'float32', 4: 'float64',
                 5: 'float128', 6: 'str', 7: 'bool', 8: 'unspecified'}
        # print('dtype',dtype[self.datatype])
        return dtype[self.datatype]

    def _get_sdf_contents(self):
        # Currently unused but kept for debugging
        for _ in range(self.nblocks):
            self.read_block_header()
            print(f'Variable={sdf.block_name}, block_type = {sdf.block_type}')

    # def check_block(self):
    #     btype = {1:'plain_mesh', 2:'point_mesh', 3:'plain_var', 4:'point_var',
    #             6:'array', 9:'stiched_tensor', 10:'stiched_material',
    #             11:'stiched_matvar', 12:'stiched_species',13:'species', 14:'plain_derived',
    #             15:'point_derived', 16:'multi_tensor', 17:'multi_material',
    #             18:'multi_matvar', 19:'multi_species'}
    #     return self.block_type in btype.keys()

# ██████  ██    ██ ████████ ███████
# ██   ██  ██  ██     ██    ██
# ██████    ████      ██    █████
# ██   ██    ██       ██    ██
# ██████     ██       ██    ███████

#  ██████  ██████  ███    ██ ██    ██ ███████ ██████  ███████ ██  ██████  ███    ██
# ██      ██    ██ ████   ██ ██    ██ ██      ██   ██ ██      ██ ██    ██ ████   ██
# ██      ██    ██ ██ ██  ██ ██    ██ █████   ██████  ███████ ██ ██    ██ ██ ██  ██
# ██      ██    ██ ██  ██ ██  ██  ██  ██      ██   ██      ██ ██ ██    ██ ██  ██ ██
#  ██████  ██████  ██   ████   ████   ███████ ██   ██ ███████ ██  ██████  ██   ████


def byte_to_int(b, split=None):
    """
    Convert a byte string to an integer

    Args:
        b (byte): byte string
        split (bool): if the byte needs split into several integers
    Returns:
        int: the integer corresponding to the byte string
        list: list of integers corresponding to the split up byte string
    """
    if split == None:
        return int.from_bytes(b, byteorder='little')
    else:
        step = len(b) // split
        return [int.from_bytes(b[i * step:i * step + step], byteorder='little') for i in range(split)]

def byte_to_float(b):
    """
    Convert a byte string to a float

    If byte string is longer than 8 bytes, the byte string is split into blocks
    of 8 bytes

    Args:
        b (byte): byte string
    Returns:
        float or list of floats: corresponding to byte string
    """

    if len(b) > 8:
        output = [unpack('d', b[8 * i:8 * i + 8])[0]
                  for i in range(int(len(b) / 8))]
    else:
        output = unpack('d', b)[0]
    return output


def byte_to_str(b):
    """
    Convert a byte string to a string

    Args:
        b (byte): byte string
    Returns:
        list: list of strings corrresponding to the byte string
    """
    out = b.decode("utf-8").replace(' ', '').split('\x00')
    return [o for o in out if o != '']  # rm trailing spaces and \x00


# ██   ██ ███████ ██      ██████  ███████ ██    ██ ██
# ██   ██ ██      ██      ██   ██ ██      ██    ██ ██
# ███████ █████   ██      ██████  █████   ██    ██ ██
# ██   ██ ██      ██      ██      ██      ██    ██ ██
# ██   ██ ███████ ███████ ██      ██       ██████  ███████

# ███████ ██    ██ ███    ██  ██████ ████████ ██  ██████  ███    ██ ███████
# ██      ██    ██ ████   ██ ██         ██    ██ ██    ██ ████   ██ ██
# █████   ██    ██ ██ ██  ██ ██         ██    ██ ██    ██ ██ ██  ██ ███████
# ██      ██    ██ ██  ██ ██ ██         ██    ██ ██    ██ ██  ██ ██      ██
# ██       ██████  ██   ████  ██████    ██    ██  ██████  ██   ████ ███████

def _rotate90cw(grid):
    s = np.shape(grid)
    new_grid = np.zeros((s[1], s[0]))
    for i in range(s[0]):
        new_grid[:, i] = grid[i, :]
    return new_grid

def _format_string(string):
    string = ' '.join(string.split('/'))
    string = ' '.join(string.split('_'))

def _unit_conv(u):
    if u in _unit_list.keys():
        return _unit_list[u]
    else:
        raise KeyError(
            f"""
            Unit {u} not fount in unit dictionary {_unit_list.keys()}
            \nWant to add it to _unit_list yourself?
            """
        )
