import xml.etree.ElementTree as ET

# Load the XML file
tree = ET.parse('../vehicle_sim/worlds/gazebo_world_description/worlds/modified_world.world')
root = tree.getroot()

# Find the world tag within sdf
world = root.find('.//world')

if world is not None:
    # Define a counter for serial naming
    counter = 1
    
    # Iterate over all model elements within the world tag
    for model in world.findall('.//model'):
        # Get the name attribute of the model
        model_name = model.get('name')
        
        # Check if 'Cone' is in the model name
        if 'Cone' in model_name:
            # Generate a new name with serial number
            new_name = f'Traffic_Cone_{counter}'
            
            # Update the model's name attribute
            model.set('name', new_name)
            
            # Increment the counter
            counter += 1

# Write the modified XML back to a file
tree.write('../vehicle_sim/worlds/gazebo_world_description/worlds/modified_world.world')