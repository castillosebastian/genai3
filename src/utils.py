import os 
import json

def string_to_json(string):
    try:
        # Convert the string to a JSON object
        json_object = json.loads(string)
        return json_object
    except json.JSONDecodeError as e:
        # Handle the exception if the string is not a valid JSON
        return f"Error converting string to JSON: {e}"
    
def get_json_field(json_string, field_name):
    # Convert the string to a JSON object
    json_object = string_to_json(json_string)

    if not isinstance(json_object, dict):
        return f"Invalid JSON"
    
    if field_name in json_object:
        # Extract the field value
        field_value = json_object[field_name]
        
        if field_value is None:
            return "Field value is null"
        return field_value
    else:
        return f"'{field_name}' key not found"