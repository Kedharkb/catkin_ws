import numpy as np


def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def find_closest_point_to_cone(points_relative_to_vehicle, cone_position, vehicle_position):
    min_distance = float('inf')
    closest_point = None
    
    for point_relative_to_vehicle in points_relative_to_vehicle:
        point_absolute = np.array(point_relative_to_vehicle) + np.array(vehicle_position)
        distance = calculate_distance(point_absolute, cone_position)
        if distance < min_distance:
            min_distance = distance
            closest_point = point_relative_to_vehicle
            
    return closest_point, min_distance

# Given data
vehicle_position = (-98.24036760305152, -201.36179774138347, 1.824735023359703)
points_relative_to_vehicle = [(-9.5, 14, -1.3), (-5.989161, 17.76643, -1.9398574)]
cone_position = (-104.199, -183.611, 0.479599)

# Find the closest point to the cone with respect to vehicle position
closest_point, distance = find_closest_point_to_cone(points_relative_to_vehicle, cone_position, vehicle_position)

print("Closest point to the cone:", closest_point)
print("Distance to cone:", distance)