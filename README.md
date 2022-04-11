### General
This is some python code that can be used to read sdf files produced by EPOCH PIC simulations. The code behaves similar to that found in the [SDF library](https://github.com/Warwick-Plasma/SDF_Matlab) for MATLAB.

### Notes
Currently only works for SDF block types of 1 and 3
This covers basically most grid variables such as:
- number density
- electric and magnetic fields
- mean kinetic energy of a a cell

The code also could be optimised I am sure

### Example Usage
```
# Initiate the class and print out contents of sdf file
sdf = ReadSDF('0001.sdf')
# Get the variable
grid, data = sdf.get_variable('ElectricField/Ey')
# Close the file (Important when looping through files!)
sdf.close_file()
```

The `get_variable` function gets the variable figures out what function needs to be called. What it returns will depend on the variable that is called. For something like Number_Density, `get_variable` will also return the associated grid with it as well unless told otherwise.

The functions like `get_plain_variable` can also be called directly using
```
sdf = ReadSDF('0001.sdf')
sdf.get_plain_variable('ElectricField/Ey')
```
but this isn't really recommended.

The `ReadSDF` class also has a function to make some plots. Instructions on how to use this will be added at some stage.

I will just be adding to this when I need to for my own simulations. Feel free to fork/make pull requests.
