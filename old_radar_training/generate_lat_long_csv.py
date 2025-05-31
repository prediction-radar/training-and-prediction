import math
import csv

def tile_to_lat_lon(x, y, z):
    """Convert tile coordinates (x, y) and zoom level z to geographic coordinates."""
    n = 2.0 ** z
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def check_tile_position(z, x, y):
    """Check if the tile's position falls within the specified ranges."""
    ranges = {
        3: {'x': (1, 2), 'y': (2, 3)},
        4: {'x': (2, 5), 'y': (5, 6)},
        5: {'x': (4,11), 'y': (10, 13)},
        6: {'x': (9, 21), 'y': (21, 28)},
        7: {'x': (19,41), 'y': (43, 55)},
        8: {'x': (39, 81), 'y': (87, 110)},
        9: {'x': (78, 161), 'y': (175, 220)},
        10: {'x': (157, 322), 'y': (350, 440)},
    }
    
    # Skip if no conditions are defined for the zoom level
    if z not in ranges:
        return False
    
    x_range = ranges[z]['x']
    y_range = ranges[z]['y']
    
    return x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]

def generate_tiles_data(zoom_levels):
    """Generate tiles data for the given range of zoom levels with specific position conditions."""
    tiles_data = []
    for z in zoom_levels:
        num_tiles = 2 ** z
        for x in range(num_tiles):
            for y in range(num_tiles):
                if check_tile_position(z, x, y):
                    top_lat, left_lon = tile_to_lat_lon(x, y, z)
                    bottom_lat, right_lon = tile_to_lat_lon(x + 1, y + 1, z)
                    tiles_data.append([z, x, y, top_lat, bottom_lat, left_lon, right_lon])
    return tiles_data

def save_to_csv(filename, data, header):
    """Save the data to a CSV file."""
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

# Define your range of zoom levels here. Adjust as needed.
zoom_levels = range(3, 11)  # Starting from 3 as specified

# Generate the tiles data
tiles_data = generate_tiles_data(zoom_levels)

# CSV header
header = ['Zoom Level', 'X', 'Y', 'Top Latitude', 'Bottom Latitude', 'Left Longitude', 'Right Longitude']

# Save the data to a CSV file
filename = 'position_filtered_tiles_data.csv'
save_to_csv(filename, tiles_data, header)

print(f"Data successfully saved to {filename}.")