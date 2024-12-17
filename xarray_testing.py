import numpy as np
import xarray as xr

num_of_x_coords = 2 
num_of_y_coords = 2

x_coords = np.arange(num_of_x_coords)  # x-coordinates: 0, 1, ..., num_of_x_coords-1
y_coords = np.arange(num_of_y_coords)  # y-coordinates: 0, 1, ..., num_of_y_coords-1

main_data = np.random.randint(0, 10, size=(num_of_x_coords, num_of_y_coords))
scalar_data = np.random.randint(0, 50, size=(num_of_x_coords, num_of_y_coords))
dataset = xr.Dataset(
    {
        "main_data": (("x", "y"), main_data),
        "scalar_data": (("x", "y"), np.array([["AB", "BC"], ["DE", "F"]])),
    },
    coords={
        "x": x_coords,
        "y": y_coords,
    }
)

print("Dataset contents:")
print(dataset)

print("\nMain data at x=0, y=0:")
print(dataset["main_data"].sel(x=0, y=0).values)

print("\nScalar value at x=0, y=0:")
print(dataset["scalar_data"].sel(x=0, y=0).values)

