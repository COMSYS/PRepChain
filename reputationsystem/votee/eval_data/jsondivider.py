import json

# Read data from "machinesensors.json" file
with open('machinesensors.json', 'r') as json_file:
    data = json.load(json_file)

# Initialize empty dictionaries for each parameter
parameter_data = {}

# Iterate through the content and populate the dictionaries
for item in data['content']:
    parameter_name = item['parameterName']
    if parameter_name not in parameter_data:
        parameter_data[parameter_name] = []

    parameter_data[parameter_name].append({
        "id": item['id'],
        "parameterName": parameter_name,
        "deviceMessageId": item['deviceMessageId'],
        "deviceTypeId": item['deviceTypeId'],
        "deviceNumber": item['deviceNumber'],
        "ioTypeId": item['ioTypeId'],
        "pinTypeId": item['pinTypeId'],
        "pinNumber": item['pinNumber'],
        "pinValue": item['pinValue'],
        "pinValueDate": item['pinValueDate']
    })

# Write the data to separate JSON files
for parameter_name, parameter_values in parameter_data.items():
    file_name = f'{parameter_name.replace(" ", "_").lower()}_data.json'
    with open(file_name, 'w') as parameter_file:
        json.dump(parameter_values, parameter_file, indent=4)
